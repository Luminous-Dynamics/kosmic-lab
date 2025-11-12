use anyhow::{bail, Context, Result};
use kosmic_holochain_tests::paths;
use tokio::process::Command;

#[cfg(feature = "holochain-stack")]
use {
    holo_hash::ActionHash,
    holochain_cli_sandbox::save,
    holochain_client::{
        AdminWebsocket, AppWebsocket, AuthorizeSigningCredentialsPayload, CellInfo,
        ClientAgentSigner, ConductorApiError, IssueAppAuthenticationTokenPayload, ZomeCallTarget,
    },
    holochain_client::{AllowedOrigins, DynAgentSigner},
    holochain_serialized_bytes::prelude::*,
    holochain_zome_types::prelude::{ExternIO, FunctionName, Record, Timestamp, ZomeName},
    serde::{Deserialize, Serialize},
};

#[cfg(feature = "holochain-stack")]
const STATE_ZOME_NAME: &str = "state_zome";

#[cfg(feature = "holochain-stack")]
const INSTALLED_APP_ID: &str = "fre-simulation";

#[cfg(feature = "holochain-stack")]
#[derive(Debug, Clone, Serialize, Deserialize, SerializedBytes)]
struct AgentStateEntryRecord {
    agent: holochain_client::AgentPubKey,
    payload: String,
    timestamp: Timestamp,
}

#[tokio::test]
async fn sandbox_script_completes() -> Result<()> {
    let repo_root = paths::repo_root();

    let status = Command::new("bash")
        .arg("-lc")
        .arg("./holochain/tests/run_sandbox.sh")
        .current_dir(&repo_root)
        .status()
        .await
        .context("failed to run sandbox harness")?;

    assert!(status.success(), "sandbox harness returned non-zero status");
    Ok(())
}

#[cfg(feature = "holochain-stack")]
fn discover_admin_port(repo_root: &std::path::Path) -> Result<u16> {
    let search_roots = [
        repo_root.join("holochain/tests"),
        repo_root.to_path_buf(),
        repo_root.join("holochain/tests/sandbox_workdir"),
    ];
    for dir in search_roots {
        let ports = save::load_ports(dir.clone()).unwrap_or_default();
        if let Some(port) = ports.into_iter().flatten().next() {
            return Ok(port);
        }
    }
    bail!("no running sandbox admin ports discovered")
}

#[cfg(feature = "holochain-stack")]
#[tokio::test]
#[ignore]
async fn state_zome_register_and_fetch() -> Result<()> {
    #[derive(Debug, Serialize)]
    struct AgentStateInputPayload<'a> {
        payload: &'a str,
    }

    let repo_root = paths::repo_root();
    let admin_port =
        discover_admin_port(&repo_root).context("failed to locate sandbox admin port")?;

    let admin = AdminWebsocket::connect(format!("127.0.0.1:{admin_port}"), None)
        .await
        .context("failed to connect to admin websocket")?;

    let app_port = match admin.list_app_interfaces().await {
        Ok(mut interfaces) if !interfaces.is_empty() => interfaces.remove(0).port,
        _ => admin
            .attach_app_interface(0, None, AllowedOrigins::Any, None)
            .await
            .context("failed to attach app interface")?,
    };

    if let Err(err) = admin.enable_app(INSTALLED_APP_ID.to_string()).await {
        if !matches!(err, ConductorApiError::ExternalApiWireError(_)) {
            return Err(anyhow::Error::msg(format!("failed to enable app: {err}")));
        }
    }

    let app_info = admin
        .list_apps(None)
        .await
        .context("failed to list apps")?
        .into_iter()
        .find(|info| info.installed_app_id == INSTALLED_APP_ID)
        .context("fre-simulation app not installed")?;

    let cell_id = app_info
        .cell_info
        .values()
        .flat_map(|cells| cells.iter())
        .find_map(|info| match info {
            CellInfo::Provisioned(cell) => Some(cell.cell_id.clone()),
            _ => None,
        })
        .context("no provisioned cell available for state zome")?;

    let credentials = admin
        .authorize_signing_credentials(AuthorizeSigningCredentialsPayload {
            cell_id: cell_id.clone(),
            functions: None,
        })
        .await
        .context("failed to authorize signing credentials")?;

    let client_signer = ClientAgentSigner::new();
    client_signer.add_credentials(cell_id.clone(), credentials);
    let signer: DynAgentSigner = client_signer.into();

    let token = admin
        .issue_app_auth_token(
            IssueAppAuthenticationTokenPayload::for_installed_app_id(INSTALLED_APP_ID.to_string())
                .expiry_seconds(0)
                .single_use(false),
        )
        .await
        .context("failed to issue app authentication token")?
        .token;

    let app_ws = AppWebsocket::connect(format!("127.0.0.1:{app_port}"), token, signer, None)
        .await
        .context("failed to connect to app websocket")?;

    let payload_value = "integration-state-payload";
    let register_io = app_ws
        .call_zome(
            ZomeCallTarget::CellId(cell_id.clone()),
            ZomeName::from(STATE_ZOME_NAME),
            FunctionName::from("register_agent_state"),
            ExternIO::encode(AgentStateInputPayload {
                payload: payload_value,
            })
            .context("failed to encode register payload")?,
        )
        .await
        .context("register_agent_state call failed")?;
    let action_hash: ActionHash = register_io
        .decode()
        .context("failed to decode action hash from register response")?;

    let get_io = app_ws
        .call_zome(
            ZomeCallTarget::CellId(cell_id.clone()),
            ZomeName::from(STATE_ZOME_NAME),
            FunctionName::from("get_agent_state"),
            ExternIO::encode(action_hash.clone())
                .context("failed to encode get_agent_state payload")?,
        )
        .await
        .context("get_agent_state call failed")?;
    let record_opt: Option<Record> = get_io
        .decode()
        .context("failed to decode get_agent_state response")?;
    let record = record_opt.context("agent state record missing")?;
    assert_eq!(record.action_address(), &action_hash);
    let entry = record
        .entry
        .to_app_option::<AgentStateEntryRecord>()
        .context("failed to deserialize agent state entry")?
        .context("agent state entry not present in record")?;
    assert_eq!(entry.payload, payload_value);

    let list_io = app_ws
        .call_zome(
            ZomeCallTarget::CellId(cell_id),
            ZomeName::from(STATE_ZOME_NAME),
            FunctionName::from("list_agent_states"),
            ExternIO::encode(app_ws.my_pub_key.clone())
                .context("failed to encode list_agent_states payload")?,
        )
        .await
        .context("list_agent_states call failed")?;
    let records: Vec<Record> = list_io
        .decode()
        .context("failed to decode list_agent_states response")?;
    assert!(
        records
            .iter()
            .any(|rec| rec.action_address() == &action_hash),
        "newly created agent state should be discoverable via list_agent_states"
    );

    Ok(())
}
