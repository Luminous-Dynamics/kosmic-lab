use hdk::prelude::*;
use std::convert::TryFrom;

#[hdk_entry_helper]
#[derive(Clone)]
pub struct AgentStateEntry {
    pub agent: AgentPubKey,
    pub payload: String,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AgentStateInput {
    pub payload: String,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
enum EntryTypes {
    #[entry_type]
    AgentStateEntry(AgentStateEntry),
}

#[hdk_link_types]
enum LinkTypes {
    AgentState,
}

#[hdk_extern]
fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

fn agent_path(agent: &AgentPubKey) -> Path {
    Path::from(format!("agent_state.{}", agent))
}

#[hdk_extern]
fn register_agent_state(input: AgentStateInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let entry = AgentStateEntry {
        agent: agent.clone(),
        payload: input.payload,
        timestamp: sys_time()?,
    };
    let action_hash = create_entry(EntryTypes::AgentStateEntry(entry))?;
    let typed_path = agent_path(&agent).typed(LinkTypes::AgentState)?;
    typed_path.ensure()?;
    create_link(
        typed_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AgentState,
        (),
    )?;
    Ok(action_hash)
}

#[hdk_extern]
fn list_agent_states(agent: AgentPubKey) -> ExternResult<Vec<Record>> {
    let typed_path = agent_path(&agent).typed(LinkTypes::AgentState)?;
    typed_path.ensure()?;
    let links = get_links(
        GetLinksInputBuilder::try_new(typed_path.path_entry_hash()?, LinkTypes::AgentState)?.build(),
    )?;
    let mut records = Vec::with_capacity(links.len());
    for link in links {
        if let Ok(action_hash) = ActionHash::try_from(link.target.clone()) {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

#[hdk_extern]
fn get_agent_state(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_contains_agent_key() {
        let bytes = vec![0u8; 32];
        let key = AgentPubKey::from_raw_32(bytes).unwrap();
        let path = agent_path(&key);
        let str_path = path.as_ref().to_string();
        assert!(str_path.contains("agent_state."));
    }
}
