pub mod client;
pub mod model;

use std::convert::Infallible;
use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    Json,
    body::Body,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;

use crate::anthropic::{
    error::json_error,
    schema::{CountTokensResponse, MessagesRequest},
};
use crate::monitor::usage_from_anthropic_sse;
use crate::provider::{CliHandlers, Provider, RequestContext};
use crate::providers::kimi::{
    count_tokens,
    translate::{
        accumulate::accumulate_response, request::translate_openai_compatible_request,
        stream::translate_stream_bytes,
    },
};

use self::client::{OpenCodeClient, OpenCodeError, OpenCodeResponse};
use self::model::EndpointKind;

enum ClientState {
    Ready(Arc<OpenCodeClient>),
    Invalid(String),
}

pub struct OpenCodeProvider {
    client: ClientState,
}

impl OpenCodeProvider {
    pub fn new() -> Self {
        let client = OpenCodeClient::new(
            crate::config::opencode_base_url(),
            crate::config::opencode_api_key(),
        )
        .map(Arc::new)
        .map(ClientState::Ready)
        .unwrap_or_else(|error| ClientState::Invalid(error.to_string()));
        Self { client }
    }

    #[cfg(test)]
    fn with_client(client: OpenCodeClient) -> Self {
        Self {
            client: ClientState::Ready(Arc::new(client)),
        }
    }

    fn client(&self) -> Result<Arc<OpenCodeClient>, String> {
        match &self.client {
            ClientState::Ready(client) => Ok(client.clone()),
            ClientState::Invalid(error) => Err(error.clone()),
        }
    }
}

impl Default for OpenCodeProvider {
    fn default() -> Self {
        Self::new()
    }
}

pub fn advertised_models() -> Vec<String> {
    model::advertised_models()
}

#[async_trait]
impl Provider for OpenCodeProvider {
    fn name(&self) -> &'static str {
        "opencode"
    }

    fn supported_models(&self) -> Vec<String> {
        advertised_models()
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &OPENCODE_CLI
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let requested = body.model.as_deref().unwrap_or_default();
        let Some(spec) = model::resolve(requested) else {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Unsupported OpenCode Go model: {requested}"),
            );
        };
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, spec.id);
            monitor.upstream_started(&ctx.req_id);
        }
        let client = match self.client() {
            Ok(client) => client,
            Err(error) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    format!("Invalid OpenCode Go configuration: {error}"),
                );
            }
        };

        match spec.endpoint {
            EndpointKind::ChatCompletions => {
                let translated = match translate_openai_compatible_request(&body, spec.id.into()) {
                    Ok(translated) => translated,
                    Err(error) => {
                        return json_error(
                            StatusCode::BAD_REQUEST,
                            "invalid_request_error",
                            error.to_string(),
                        );
                    }
                };
                let upstream = match client
                    .post(spec.endpoint, &translated, true, ctx.traffic.clone())
                    .await
                {
                    Ok(upstream) => upstream,
                    Err(error) => return map_error(error),
                };
                chat_response(upstream, &body, requested, &ctx).await
            }
            EndpointKind::Messages => {
                let mut translated = match serde_json::to_value(&body) {
                    Ok(value) => value,
                    Err(error) => {
                        return json_error(
                            StatusCode::BAD_REQUEST,
                            "invalid_request_error",
                            error.to_string(),
                        );
                    }
                };
                translated["model"] = serde_json::Value::String(spec.id.to_string());
                let upstream = match client
                    .post(spec.endpoint, &translated, body.stream, ctx.traffic.clone())
                    .await
                {
                    Ok(upstream) => upstream,
                    Err(error) => return map_error(error),
                };
                messages_response(upstream, body.stream, &ctx).await
            }
        }
    }

    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let requested = body.model.as_deref().unwrap_or_default();
        let Some(spec) = model::resolve(requested) else {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Unsupported OpenCode Go model: {requested}"),
            );
        };
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, spec.id);
        }
        let tokens = count_tokens::count_tokens(&body);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.usage_updated(&ctx.req_id, Some(tokens), None);
        }
        (
            StatusCode::OK,
            Json(CountTokensResponse {
                input_tokens: tokens,
            }),
        )
            .into_response()
    }
}

async fn chat_response(
    upstream: OpenCodeResponse,
    body: &MessagesRequest,
    requested: &str,
    ctx: &RequestContext,
) -> Response {
    let bytes = match upstream.into_bytes().await {
        Ok(bytes) => bytes,
        Err(error) => return map_error(error),
    };
    if let Some(traffic) = ctx.traffic.as_ref() {
        traffic.write_bytes("032-upstream-response-body.sse", &bytes);
    }
    let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    if body.stream {
        let translated = match translate_stream_bytes(&bytes, &message_id, requested) {
            Ok(translated) => translated,
            Err(error) => {
                return json_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    format!("OpenCode Go stream translation failed: {error}"),
                );
            }
        };
        if let Some(monitor) = ctx.monitor.as_ref() {
            let (input_tokens, output_tokens) = usage_from_anthropic_sse(&translated);
            monitor.stream_progress(
                &ctx.req_id,
                translated.len() as u64,
                count_sse_events(&translated),
                input_tokens,
                output_tokens,
            );
        }
        (
            [
                (http::header::CONTENT_TYPE, "text/event-stream"),
                (http::header::CACHE_CONTROL, "no-cache"),
                (http::header::CONNECTION, "keep-alive"),
            ],
            translated,
        )
            .into_response()
    } else {
        match accumulate_response(&bytes, &message_id, requested) {
            Ok(value) => {
                if let Some(monitor) = ctx.monitor.as_ref() {
                    monitor.usage_updated(
                        &ctx.req_id,
                        value
                            .pointer("/usage/input_tokens")
                            .and_then(|value| value.as_u64()),
                        value
                            .pointer("/usage/output_tokens")
                            .and_then(|value| value.as_u64()),
                    );
                }
                (StatusCode::OK, Json(value)).into_response()
            }
            Err(error) => json_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("OpenCode Go response translation failed: {error}"),
            ),
        }
    }
}

async fn messages_response(
    upstream: OpenCodeResponse,
    stream: bool,
    ctx: &RequestContext,
) -> Response {
    if stream {
        let monitor = ctx.monitor.clone();
        let req_id = ctx.req_id.clone();
        let mut bytes = 0u64;
        let mut chunks = 0u64;
        let stream = upstream.into_stream().map(move |chunk| {
            let output = match chunk {
                Ok(chunk) => {
                    bytes = bytes.saturating_add(chunk.len() as u64);
                    chunks = chunks.saturating_add(1);
                    if let Some(monitor) = monitor.as_ref() {
                        let (input_tokens, output_tokens) = usage_from_anthropic_sse(&chunk);
                        monitor.stream_progress(
                            &req_id,
                            bytes,
                            chunks,
                            input_tokens,
                            output_tokens,
                        );
                    }
                    chunk
                }
                Err(_) => Bytes::from_static(
                    b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"OpenCode Go upstream stream failed\"}}\n\n",
                ),
            };
            Ok::<Bytes, Infallible>(output)
        });
        (
            [
                (http::header::CONTENT_TYPE, "text/event-stream"),
                (http::header::CACHE_CONTROL, "no-cache"),
                (http::header::CONNECTION, "keep-alive"),
            ],
            Body::from_stream(stream),
        )
            .into_response()
    } else {
        let bytes = match upstream.into_bytes().await {
            Ok(bytes) => bytes,
            Err(error) => return map_error(error),
        };
        if let Some(traffic) = ctx.traffic.as_ref() {
            traffic.write_bytes("032-upstream-response-body.json", &bytes);
        }
        let value = match serde_json::from_slice::<serde_json::Value>(&bytes) {
            Ok(value) => value,
            Err(_) => {
                return json_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    "OpenCode Go returned invalid JSON",
                );
            }
        };
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.usage_updated(
                &ctx.req_id,
                value
                    .pointer("/usage/input_tokens")
                    .and_then(|value| value.as_u64()),
                value
                    .pointer("/usage/output_tokens")
                    .and_then(|value| value.as_u64()),
            );
        }
        (StatusCode::OK, Json(value)).into_response()
    }
}

fn map_error(error: OpenCodeError) -> Response {
    let error_type = match error.status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => "authentication_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        status if status.is_client_error() => "invalid_request_error",
        _ => "api_error",
    };
    let status = if error.status.is_server_error() {
        StatusCode::BAD_GATEWAY
    } else {
        error.status
    };
    let response = json_error(status, error_type, error.message);
    if let Some(retry_after) = error.retry_after {
        ([(http::header::RETRY_AFTER, retry_after)], response).into_response()
    } else {
        response
    }
}

fn count_sse_events(bytes: &[u8]) -> u64 {
    String::from_utf8_lossy(bytes).matches("event:").count() as u64
}

struct OpenCodeCli;

impl CliHandlers for OpenCodeCli {
    fn login(&self) -> anyhow::Result<()> {
        anyhow::bail!(
            "OpenCode Go uses an API key; set CCP_OPENCODE_API_KEY, OPENCODE_API_KEY, or opencode.apiKey in config.json"
        )
    }

    fn device(&self) -> anyhow::Result<()> {
        self.login()
    }

    fn status(&self) -> anyhow::Result<()> {
        let Some(source) = crate::config::opencode_api_key_source() else {
            anyhow::bail!("Not authenticated");
        };
        println!("API key configured: true");
        println!("Source: {source}");
        println!("Base URL: {}", crate::config::opencode_base_url());
        Ok(())
    }

    fn logout(&self) -> anyhow::Result<()> {
        anyhow::bail!(
            "OpenCode Go credentials are managed through environment variables or config.json"
        )
    }
}

static OPENCODE_CLI: OpenCodeCli = OpenCodeCli;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::RequestContext;
    use axum::{Json, Router, extract::OriginalUri, http::HeaderMap, routing::post};
    use serde_json::json;

    fn context() -> RequestContext {
        RequestContext {
            req_id: "req_test".to_string(),
            provider: "opencode".to_string(),
            session_id: None,
            session_seq: None,
            monitor: None,
            traffic: None,
        }
    }

    #[tokio::test]
    async fn missing_key_is_actionable() {
        let client = OpenCodeClient::new("https://example.com/v1".into(), None).unwrap();
        let provider = OpenCodeProvider::with_client(client);
        let body: MessagesRequest = serde_json::from_value(json!({
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        let response = provider.handle_messages(body, context()).await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("OPENCODE_API_KEY")
        );
    }

    #[tokio::test]
    async fn count_tokens_is_local_for_both_protocol_families() {
        let provider = OpenCodeProvider::with_client(
            OpenCodeClient::new("https://example.com/v1".into(), None).unwrap(),
        );
        for model in ["glm-5.2", "minimax-m3"] {
            let body: MessagesRequest = serde_json::from_value(json!({
                "model": model,
                "messages": [{"role": "user", "content": "hello world"}]
            }))
            .unwrap();
            let response = provider.handle_count_tokens(body, context()).await;
            assert_eq!(response.status(), StatusCode::OK);
        }
    }

    async fn mock_go_upstream(
        OriginalUri(uri): OriginalUri,
        headers: HeaderMap,
        Json(body): Json<serde_json::Value>,
    ) -> Response {
        if uri.path().ends_with("/chat/completions") {
            assert_eq!(
                headers
                    .get(http::header::AUTHORIZATION)
                    .and_then(|value| value.to_str().ok()),
                Some("Bearer test-key")
            );
        } else {
            assert_eq!(
                headers
                    .get("x-api-key")
                    .and_then(|value| value.to_str().ok()),
                Some("test-key")
            );
        }
        if body["model"] == "qwen3.7-max" {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                [(http::header::RETRY_AFTER, "17")],
                Json(json!({"error":{"message":"Go limit reached"}})),
            )
                .into_response();
        }
        if uri.path().ends_with("/chat/completions") {
            return (
                [(http::header::CONTENT_TYPE, "text/event-stream")],
                concat!(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"hello from chat\"}}]}\n\n",
                    "data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n\n",
                    "data: [DONE]\n\n"
                ),
            )
                .into_response();
        }
        if body["stream"] == true {
            return (
                [(http::header::CONTENT_TYPE, "text/event-stream")],
                concat!(
                    "event: message_start\n",
                    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_native\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"minimax-m3\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":4,\"output_tokens\":0}}}\n\n",
                    "event: message_stop\n",
                    "data: {\"type\":\"message_stop\"}\n\n"
                ),
            )
                .into_response();
        }
        Json(json!({
            "id": "msg_native",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [{"type":"text","text":"hello from messages"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens":4,"output_tokens":3}
        }))
        .into_response()
    }

    async fn mock_provider() -> (OpenCodeProvider, tokio::task::JoinHandle<()>) {
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_go_upstream))
            .route("/v1/messages", post(mock_go_upstream));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client =
            OpenCodeClient::new(format!("http://{address}/v1"), Some("test-key".to_string()))
                .unwrap();
        (OpenCodeProvider::with_client(client), server)
    }

    #[tokio::test]
    async fn provider_translates_chat_and_passes_messages_through() {
        let (provider, server) = mock_provider().await;

        let chat: MessagesRequest = serde_json::from_value(json!({
            "model": "glm-5.2",
            "stream": false,
            "messages": [{"role":"user","content":"hello"}]
        }))
        .unwrap();
        let response = provider.handle_messages(chat, context()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["content"][0]["text"], "hello from chat");

        let messages: MessagesRequest = serde_json::from_value(json!({
            "model": "opencode-go/minimax-m3",
            "stream": false,
            "messages": [{"role":"user","content":"hello"}]
        }))
        .unwrap();
        let response = provider.handle_messages(messages, context()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["model"], "minimax-m3");
        assert_eq!(value["content"][0]["text"], "hello from messages");
        server.abort();
    }

    #[tokio::test]
    async fn messages_stream_is_passthrough_and_rate_limit_is_preserved() {
        let (provider, server) = mock_provider().await;
        let streaming: MessagesRequest = serde_json::from_value(json!({
            "model": "minimax-m3",
            "stream": true,
            "messages": [{"role":"user","content":"hello"}]
        }))
        .unwrap();
        let response = provider.handle_messages(streaming, context()).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("event: message_stop"));

        let limited: MessagesRequest = serde_json::from_value(json!({
            "model": "qwen3.7-max",
            "messages": [{"role":"user","content":"hello"}]
        }))
        .unwrap();
        let response = provider.handle_messages(limited, context()).await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get(http::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("17")
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["error"]["type"], "rate_limit_error");
        assert_eq!(value["error"]["message"], "Go limit reached");
        server.abort();
    }
}
