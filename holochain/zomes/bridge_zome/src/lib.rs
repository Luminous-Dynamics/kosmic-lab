use hdk::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeRequest {
    pub peer: AgentPubKey,
    pub window: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeResponse {
    pub te_mutual: f64,
    pub te_symmetry: f64,
}

#[hdk_extern]
fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

#[hdk_extern]
fn request_bridge(_req: BridgeRequest) -> ExternResult<BridgeResponse> {
    // Placeholder: real implementation will compute TE from shared histories.
    let response = BridgeResponse {
        te_mutual: 0.0,
        te_symmetry: 0.0,
    };
    Ok(response)
}
