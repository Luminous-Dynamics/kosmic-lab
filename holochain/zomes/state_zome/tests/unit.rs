#[cfg(test)]
mod state_tests {
    use super::super::*;

    #[test]
    fn agent_path_contains_prefix() {
        let key_bytes = vec![0u8; 32];
        let key = AgentPubKey::from_raw_32(key_bytes).unwrap();
        let path = super::agent_path(&key);
        assert!(path.as_ref().to_string().starts_with("agent_state."));
    }
}
