pub const MODEL_PREFIX: &str = "opencode-go/";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointKind {
    ChatCompletions,
    Messages,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    pub id: &'static str,
    pub endpoint: EndpointKind,
}

pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "glm-5.2",
        endpoint: EndpointKind::ChatCompletions,
    },
    ModelSpec {
        id: "glm-5.1",
        endpoint: EndpointKind::ChatCompletions,
    },
    ModelSpec {
        id: "glm-5",
        endpoint: EndpointKind::ChatCompletions,
    },
    ModelSpec {
        id: "kimi-k2.7-code",
        endpoint: EndpointKind::ChatCompletions,
    },
    ModelSpec {
        id: "kimi-k2.6",
        endpoint: EndpointKind::ChatCompletions,
    },
    ModelSpec {
        id: "kimi-k2.5",
        endpoint: EndpointKind::ChatCompletions,
    },
    ModelSpec {
        id: "deepseek-v4-pro",
        endpoint: EndpointKind::ChatCompletions,
    },
    ModelSpec {
        id: "deepseek-v4-flash",
        endpoint: EndpointKind::ChatCompletions,
    },
    ModelSpec {
        id: "mimo-v2.5",
        endpoint: EndpointKind::ChatCompletions,
    },
    ModelSpec {
        id: "mimo-v2.5-pro",
        endpoint: EndpointKind::ChatCompletions,
    },
    ModelSpec {
        id: "qwen3.5-plus",
        endpoint: EndpointKind::ChatCompletions,
    },
    ModelSpec {
        id: "minimax-m3",
        endpoint: EndpointKind::Messages,
    },
    ModelSpec {
        id: "minimax-m2.7",
        endpoint: EndpointKind::Messages,
    },
    ModelSpec {
        id: "minimax-m2.5",
        endpoint: EndpointKind::Messages,
    },
    ModelSpec {
        id: "qwen3.7-max",
        endpoint: EndpointKind::Messages,
    },
    ModelSpec {
        id: "qwen3.7-plus",
        endpoint: EndpointKind::Messages,
    },
    ModelSpec {
        id: "qwen3.6-plus",
        endpoint: EndpointKind::Messages,
    },
];

pub fn resolve(raw: &str) -> Option<ModelSpec> {
    let id = raw.strip_prefix(MODEL_PREFIX).unwrap_or(raw);
    MODELS.iter().copied().find(|model| model.id == id)
}

pub fn advertised_models() -> Vec<String> {
    let mut result = Vec::with_capacity(MODELS.len() * 2 - 1);
    for model in MODELS {
        // `kimi-k2.6` is already a public alias of the Kimi Code provider.
        // Keep that behavior stable and expose the Go model via its canonical,
        // provider-qualified OpenCode id.
        if model.id != "kimi-k2.6" {
            result.push(model.id.to_string());
        }
        result.push(format!("{MODEL_PREFIX}{}", model.id));
    }
    result.sort_unstable();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_catalog_is_partitioned_by_wire_protocol() {
        let chat = MODELS
            .iter()
            .filter(|model| model.endpoint == EndpointKind::ChatCompletions)
            .count();
        let messages = MODELS
            .iter()
            .filter(|model| model.endpoint == EndpointKind::Messages)
            .count();
        assert_eq!(chat, 11);
        assert_eq!(messages, 6);
    }

    #[test]
    fn canonical_prefix_resolves_and_unknown_models_do_not() {
        let spec = resolve("opencode-go/minimax-m3").expect("known model");
        assert_eq!(spec.id, "minimax-m3");
        assert_eq!(spec.endpoint, EndpointKind::Messages);
        assert!(resolve("opencode-go/not-a-model").is_none());
    }

    #[test]
    fn conflicting_kimi_id_is_only_advertised_with_prefix() {
        let models = advertised_models();
        assert!(!models.iter().any(|model| model == "kimi-k2.6"));
        assert!(models.iter().any(|model| model == "opencode-go/kimi-k2.6"));
    }
}
