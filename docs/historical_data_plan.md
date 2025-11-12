# Historical K(t) Data Expansion Plan

Current proxies (OWID energy + World Bank trade/migration/remittances) provide basic coverage. Next targets:

| Harmony | Candidate dataset | Source |
|---------|-------------------|--------|
| Interconnection | International trade volume by region (IMF DOTS / UN Comtrade) | UN Comtrade API |
| Reciprocity | Global aid vs. extraction flows (OECD DAC) | https://stats.oecd.org |
| Play entropy | Occupational diversity indices (Clio Infra) | https://www.clio-infra.eu *(access requires dataset request; placeholder not substituted)* |
| Wisdom | Global patent counts by technology (WIPO) | https://www3.wipo.int *(public download currently unavailable; see notes)* |
| Flourishing | Mortality/life expectancy (UN DESA, Gapminder) | https://population.un.org |

Action items:
1. Identify open APIs/downloads, convert to decadal series (similar to OWID/World Bank scripts).
2. Extend `historical_k/data` scripts with fetch & normalization logic.
3. Update `historical_k/k_config.yaml` and rerun pipeline once each dataset integrated.
4. Record hashes/sources in `docs/data_sources.md`.

## Notes
- OWID/WIPO patent downloads (`patent-applications.csv`, `Applications_by_residence.csv.zip`) returned HTTP 404 / invalid ZIP as of the latest attempt. Investigate alternative mirrors or request archived data from WIPO before scripting ingestion.
- Clio Infra occupational diversity dataset is not directly downloadable; pause ingestion until a licensed copy is obtained.
