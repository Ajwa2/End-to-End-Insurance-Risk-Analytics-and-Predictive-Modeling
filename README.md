# AlphaCare Insurance Analytics

Project: AlphaCare Insurance Analytics — historical car insurance claims analysis (Feb 2014 — Aug 2015) for South Africa.

Background
 - AlphaCare Insurance Solutions (ACIS) wants to improve marketing precision by identifying low-risk policyholders and improving profitability. This repository contains the code and notebooks for exploratory data analysis, statistical testing, and baseline machine learning models focused on risk (claims) and profitability (premiums).

Data Description
 - The primary dataset is provided in `data/MachineLearningRating_v3.txt` (pipe-separated). Key schema fields include:
	 - Policy: `UnderwrittenCoverID`, `PolicyID`, `TransactionMonth`
	 - Client: `IsVATRegistered`, `Citizenship`, `LegalType`, `Title`, `Language`, `Bank`, `AccountType`, `MaritalStatus`, `Gender`
	 - Location: `Country`, `Province`, `PostalCode`, `MainCrestaZone`, `SubCrestaZone`
	 - Car: `ItemType`, `mmcode`, `VehicleType`, `RegistrationYear`, `Make`, `Model`, `Cylinders`, `cubiccapacity`, `kilowatts`, `Bodytype`, `NumberOfDoors`, `VehicleIntroDate`, `CustomValueEstimate`, `AlarmImmobiliser`, `TrackingDevice`, `CapitalOutstanding`, `NewVehicle`, `WrittenOff`, `Rebuilt`, `Converted`, `CrossBorder`, `NumberOfVehiclesInFleet`
	 - Plan: `SumInsured`, `TermFrequency`, `CalculatedPremiumPerTerm`, `ExcessSelected`, `CoverCategory`, `CoverType`, `CoverGroup`, `Section`, `Product`, `StatutoryClass`, `StatutoryRiskType`
	 - Financials: `TotalPremium`, `TotalClaims`

Goals
 - EDA: Understand distributions of premiums and claims, compute key KPIs such as Loss Ratio (TotalClaims / TotalPremium), and identify data quality issues.
 - Hypothesis testing: Statistically evaluate which factors (vehicle type, age, location, cover type, etc.) are associated with claim frequency/severity.
 - Machine Learning: Build baseline models to predict claim amount or claim occurrence for targeted marketing and risk-based pricing.

Repository layout
 - `.github/workflows/` — CI workflows (unit tests, linters)
 - `data/` — raw and processed data artifacts (do not commit large raw data if possible)
 - `notebooks/` — exploratory notebooks (EDA and modeling)
 - `src/` — Python scripts and modules for data processing and analysis
 - `tests/` — unit tests
 - `scripts/` — utility scripts and helpers

Getting started
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook notebooks/01_EDA.ipynb
```

Contact
 - For questions about the analysis or datasets contact the analytics team lead at ACIS.
