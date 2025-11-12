use hdk::prelude::*;
use std::convert::{TryFrom, TryInto};

#[hdk_entry_helper]
#[derive(Clone)]
pub struct HarmonyMetricsEntry {
    pub k_index: f64,
    pub harmonies: [f64; 7],
    pub timestamp: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type]
    HarmonyMetricsEntry(HarmonyMetricsEntry),
}

#[hdk_link_types]
pub enum LinkTypes {
    MetricsByWindow,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MetricsWindowQuery {
    pub start: Timestamp,
    pub end: Timestamp,
}

#[hdk_extern]
fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

fn window_bucket(ts: &Timestamp) -> i64 {
    (ts.as_micros() / 1_000_000 / 3600) as i64
}

#[hdk_extern]
fn publish_metrics(entry: HarmonyMetricsEntry) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::HarmonyMetricsEntry(entry.clone()))?;
    let typed_path = Path::from(format!("metrics_window.{}", window_bucket(&entry.timestamp)))
        .typed(LinkTypes::MetricsByWindow)?;
    typed_path.ensure()?;
    create_link(
        typed_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::MetricsByWindow,
        (),
    )?;
    Ok(action_hash)
}

#[hdk_extern]
fn list_metrics(_: ()) -> ExternResult<Vec<Record>> {
    let entry_type: EntryType = UnitEntryTypes::HarmonyMetricsEntry.try_into()?;

    let filter = ChainQueryFilter::new().entry_type(entry_type).include_entries(true);
    query(filter)
}

#[hdk_extern]
fn query_metrics_window(range: MetricsWindowQuery) -> ExternResult<Vec<Record>> {
    if range.end < range.start {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    let start_bucket = window_bucket(&range.start);
    let end_bucket = window_bucket(&range.end);

    for bucket in start_bucket..=end_bucket {
        let typed_path = Path::from(format!("metrics_window.{}", bucket))
            .typed(LinkTypes::MetricsByWindow)?;
        typed_path.ensure()?;
        let links = get_links(
            GetLinksInputBuilder::try_new(
                typed_path.path_entry_hash()?,
                LinkTypes::MetricsByWindow,
            )?
            .build(),
        )?;
        for link in links {
            if let Ok(action_hash) = ActionHash::try_from(link.target.clone()) {
                if let Some(record) = get(action_hash, GetOptions::default())? {
                    records.push(record);
                }
            }
        }
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_monotonic() {
        let base = Timestamp::from_micros(0);
        let later = Timestamp::from_micros(3_600_000_000);
        assert!(window_bucket(&later) > window_bucket(&base));
    }
}
