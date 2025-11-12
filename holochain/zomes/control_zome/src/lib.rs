use hdk::prelude::*;
use std::convert::TryFrom;

#[hdk_entry_helper]
#[derive(Clone)]
pub struct KnobUpdateEntry {
    pub communication_cost: f64,
    pub plasticity_rate: f64,
    pub stimulus_intensity: f64,
    pub timestamp: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
enum EntryTypes {
    #[entry_type]
    KnobUpdateEntry(KnobUpdateEntry),
}

#[hdk_link_types]
enum LinkTypes {
    KnobUpdates,
}

#[hdk_extern]
fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KnobUpdateInput {
    pub communication_cost: f64,
    pub plasticity_rate: f64,
    pub stimulus_intensity: f64,
}

#[hdk_extern]
fn broadcast_knob(input: KnobUpdateInput) -> ExternResult<ActionHash> {
    let entry = KnobUpdateEntry {
        communication_cost: input.communication_cost,
        plasticity_rate: input.plasticity_rate,
        stimulus_intensity: input.stimulus_intensity,
        timestamp: sys_time()?,
    };
    let hash = create_entry(EntryTypes::KnobUpdateEntry(entry.clone()))?;
    let typed_path = Path::from("knob_updates").typed(LinkTypes::KnobUpdates)?;
    typed_path.ensure()?;
    create_link(
        typed_path.path_entry_hash()?,
        hash.clone(),
        LinkTypes::KnobUpdates,
        (),
    )?;
    emit_signal(&entry)?;
    Ok(hash)
}

#[hdk_extern]
fn list_knob_updates(_: ()) -> ExternResult<Vec<Record>> {
    let typed_path = Path::from("knob_updates").typed(LinkTypes::KnobUpdates)?;
    typed_path.ensure()?;
    let links = get_links(
        GetLinksInputBuilder::try_new(typed_path.path_entry_hash()?, LinkTypes::KnobUpdates)?.build(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_input_within_bounds() {
        let input = KnobUpdateInput {
            communication_cost: 0.5,
            plasticity_rate: 0.5,
            stimulus_intensity: 1.0,
        };
        assert!((0.0..=1.0).contains(&input.communication_cost));
        assert!(input.stimulus_intensity >= 0.0);
    }
}
