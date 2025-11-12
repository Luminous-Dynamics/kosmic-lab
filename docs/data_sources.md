# Historical & Simulation Data Sources

Document each dataset used in the project with provenance and integrity hashes.

| Dataset | Description | Source / URL | SHA256 |
|---------|-------------|--------------|--------|
| `historical_k/data/OWID_energy.csv` | Our World In Data global energy dataset | https://ourworldindata.org/energy | `04cd66dc71a96fdae02f5c379d00976e0d9644df807898085489e4a9d082c21f` |
| `historical_k/data/network_modularity_inverse.csv` | Derived proxy (electricity demand) | `build_from_owid.py` | `b4dfb90197b75de930b891fdbca24ad30adf9989a52eddac9cb3da0b3825a645` |
| `historical_k/data/communication_latency_inverse.csv` | Derived proxy (1 - carbon intensity) | `build_from_owid.py` | `62b2c59bd162232ec3771bf9ba2355cbb3ed24005b05b5b9368f02b1b43748dd` |
| `historical_k/data/trade_network_degree.csv` | Derived proxy (GDP) | `build_from_owid.py` | `c411752fdac68e01940bf1400e499872e4b619c2c4fd20550a103ff44ffebc87` |
| `historical_k/data/migration_flux_index.csv` | Derived proxy (population change) | `build_from_owid.py` | `19c06d1b94338334c6d9f090dbdf84e3302f3d4566930bf85b558b83a81beffe` |
| `historical_k/data/bilateral_trade_symmetry.csv` | Derived proxy (low-carbon electricity) | `build_from_owid.py` | `baf635231563477becc533a752ae0c9ba2698f2de8adb02ff58aea302576001d` |
| `historical_k/data/alliance_reciprocity_ratio.csv` | Derived proxy (biofuel share) | `build_from_owid.py` | `50d464ee2ee3e9f567f81198f494fe7a534d1c1a35d6d05abd60c65ca9c1cacc` |
| `historical_k/data/occupation_diversity_entropy.csv` | Derived proxy (renewables share) | `build_from_owid.py` | `94f2e0abc6e20261a6dc938595f63c9bcf7cca3713422561130579cb29e3c0a9` |
| `historical_k/data/innovation_field_entropy.csv` | Derived proxy (energy per capita) | `build_from_owid.py` | `0c227b57171e47c2a5de43bd55cdb41c4dc9afe263b3c2a670246b3087f62d11` |
| `historical_k/data/forecast_skill_index.csv` | Derived proxy (energy per GDP, inverted) | `build_from_owid.py` | `3281bc4fb96e4b1d30918b9a731a7175891af0ae7752c52f0d7b50420cab02f2` |
| `historical_k/data/error_correction_speed.csv` | Derived proxy (energy consumption volatility) | `build_from_owid.py` | `775d940e3da76156990387d6f4c2f2225de2a86291419faa89584c327940ee82` |
| `historical_k/data/trade_share_gdp.csv` | World Bank NE.TRD.GNFS.ZS (trade % GDP) | World Bank API | `731e4856fa078b977d2f8c491072f0b438344592680229719b6c29be57b817eb` |
| `historical_k/data/net_migration.csv` | World Bank SM.POP.NETM (net migration) | World Bank API | `ccf6da06e35e84f846f2deb83dacf46c357b2f121cd45f8adafe33d68d546516` |
| `historical_k/data/remittance_inflows.csv` | World Bank BX.TRF.PWKR.CD.DT (remittances, US$) | World Bank API | `0a73e108d5ec15974523071ab094fcbdd5ad079aebda969576cb9e6d86a3b5a1` |
| `historical_k/data/aid_extraction_balance.csv` | Aid vs. extraction proxy = ODA / FDI outflows | World Bank API (DT.ODA.ODAT.CD, BM.KLT.DINV.CD.WD) | `4adee449a853649b18b6bd18a8d03c0a71213c8dbe6d113afab3aa813ccf6414` |
| `historical_k/data/patent_resident_filings.csv` | Patent applications, residents | World Bank API (IP.PAT.RESD) | `bce7785e8176f3afb9808b07648aa1620b27c3ee6c57d6381c9facb772e4fdc6` |
| `historical_k/data/global_trade_volume.csv` | Imports + exports of goods/services (US$) | World Bank API (NE.IMP.GNFS.CD, NE.EXP.GNFS.CD) | `318660a59a0da05198531012a17dfe504b9075bbbbab360c719a3866100911f4` |
| `historical_k/data/life_expectancy_global.csv` | Period life expectancy at birth (World) | Our World In Data (life-expectancy.csv) | `1d4bc043bd1aae21a6e669929c88c91351438a65e3ac5b593728c0c8131118f8` |
| `historical_k/data/income_fairness_index.csv` | Global mean Gini transformed to fairness | World Bank API (SI.POV.GINI averaged across all countries) | `3a52bde58ad1eac5ce68109f4b87c94013aa0e5ef937e3bece204dece88b0240` |
| `historical_k/data/education_enrolment_index.csv` | Normalized mean school-life expectancy | World Bank API (SE.SCH.LIFE, averaged across countries) | `0e69482407e9858e300a640fbacb3ff8fef1bea40be1bba31a2025d32b52a3f5` |
| `historical_k/data/environmental_performance_index.csv` | Inverted electricity use per capita | World Bank API (EG.USE.ELEC.KH.PC averaged across countries) | `f7d65161da8a9ace7958b079b5b05f5c22026921dbf1592a45236f213433bfbe` |
| `historical_k/data/renewable_energy_share.csv` | Normalized renewable energy consumption share | World Bank API (EG.FEC.RNEW.ZS averaged across countries) | `376e881ef9a9cceb230cf0f84ce51ba6ca9ac33925a92d94efd674185bcf6520` |
| `historical_k/data/research_spending_index.csv` | Normalized R&D expenditure (% GDP) | World Bank API (GB.XPD.RSDV.GD.ZS averaged across countries) | `844e3f04e6779fb639646b4d8cce3b0b7808a8f513063417b035361fbef5dd13` |
| `historical_k/data/trade_openness_index.csv` | Normalized trade openness (% of GDP) | World Bank API (NE.TRD.GNFS.ZS averaged across countries) | `08524e633ddfabb915088a362d4120f00bf8c3fa49d7b42380ca1db2ce1b9254` |

Please update this table as real datasets replace the synthetic defaults.
