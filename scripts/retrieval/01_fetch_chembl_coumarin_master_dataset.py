#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ChEMBL Coumarin Retrieval and Metadata Assembly Script
- Parallel fetching + checkpoint resume
- Activities + Molecules(SMILES/InChIKey) + Assays + Targets(UniProt) + Documents(DOI)
- Optional RDKit chemistry descriptors + Murcko scaffold
- Outputs a single master CSV (and optional Parquet if you want)

Requirements:
  pip install pandas requests tqdm numpy rdkit-pypi pyarrow

Input:
  ChEMBL_ID.txt  (one CHEMBLxxxxx per line)

Output:
  full_scale_results_2026/coumarin_master_YYYYMMDD_HHMMSS.csv
  full_scale_results_2026/summary_YYYYMMDD_HHMMSS.json
"""

import os
import time
import json
import math
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# --- Optional RDKit block (set USE_RDKIT=False if you don't want it)
USE_RDKIT = True
try:
    if USE_RDKIT:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:
    USE_RDKIT = False


class ChEMBLMasterScanner:
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CoumarinMasterScanner/2026",
            "Accept": "application/json"
        })

        # --- Performance / API etiquette
        self.MAX_WORKERS = 12
        self.BATCH_SIZE = 75
        self.TIMEOUT = 45
        self.MAX_RETRIES = 4
        self.SOFT_SLEEP = 0.05  # small jitter to reduce burst

        # --- Output dirs
        self.output_dir = "full_scale_results_2026"
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- Stats
        self.stats = {
            "total_processed_molecules": 0,
            "successful_molecules": 0,
            "failed_molecules": 0,
            "total_activities": 0,
        }

        # --- Metadata caches
        self.cache = {
            "molecule": {},
            "assay": {},
            "target": {},
            "document": {},
        }
        self.cache_lock = threading.Lock()

    # ---------------------------
    # Low-level GET with retry
    # ---------------------------
    def get_json(self, url, params=None):
        for attempt in range(self.MAX_RETRIES):
            try:
                r = self.session.get(url, params=params, timeout=self.TIMEOUT)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                # transient errors
                time.sleep(1 + attempt * 0.5)
            except Exception:
                time.sleep(1 + attempt * 0.5)
        return None

    # ---------------------------
    # Phase-1: Fetch activities per molecule
    # ---------------------------
    def fetch_molecule_activities(self, chembl_id):
        url = f"{self.base_url}/activity.json"
        params = {
            "molecule_chembl_id": chembl_id,
            "limit": 1000,
            "order_by": "-pchembl_value"
        }

        data = self.get_json(url, params=params)
        if not data:
            return [], {"status": "no_response", "id": chembl_id}

        activities = data.get("activities", [])
        if not activities:
            return [], {"status": "no_activities", "id": chembl_id}

        processed = []
        for act in activities:
            if not isinstance(act, dict):
                continue

            # only keep records with some measurable quantity
            # (you can relax/tighten later)
            if act.get("standard_value") is None and act.get("pchembl_value") is None:
                continue

            rec = {
                # keys for joins
                "molecule_chembl_id": chembl_id,
                "activity_id": act.get("activity_id", ""),
                "assay_chembl_id": act.get("assay_chembl_id", ""),
                "target_chembl_id": act.get("target_chembl_id", ""),
                "document_chembl_id": act.get("document_chembl_id", ""),

                # target quick fields (often present, but we will overwrite/standardize via target endpoint)
                "target_pref_name": act.get("target_pref_name", ""),
                "target_type": act.get("target_type", ""),
                "target_organism": act.get("target_organism", ""),

                # activity core fields (Tier-A / censoring)
                "standard_type": act.get("standard_type", ""),
                "standard_relation": act.get("standard_relation", ""),
                "standard_value": act.get("standard_value", None),
                "standard_units": act.get("standard_units", ""),
                "standard_flag": act.get("standard_flag", ""),
                "standard_text_value": act.get("standard_text_value", ""),

                # model-friendly
                "pchembl_value": act.get("pchembl_value", None),

                # traceability / QA
                "data_validity_comment": act.get("data_validity_comment", ""),
                "activity_comment": act.get("activity_comment", ""),
                "src_id": act.get("src_id", ""),
                "record_id": act.get("record_id", ""),

                # optional BAO (may be absent here; we also pull from assay endpoint)
                "bao_label": act.get("bao_label", ""),
                "bao_format": act.get("bao_format", ""),

                "fetch_time": datetime.now().isoformat(timespec="seconds"),
            }
            processed.append(rec)

        return processed, {"status": "success", "id": chembl_id, "count": len(processed)}

    # ---------------------------
    # Checkpoint helpers
    # ---------------------------
    def find_last_checkpoint(self):
        files = [f for f in os.listdir(self.checkpoint_dir)
                 if f.startswith("batch_") and f.endswith(".csv")]
        if not files:
            return 0
        mx = 0
        for f in files:
            try:
                num = int(f.split("_")[1].split(".")[0])
                mx = max(mx, num)
            except Exception:
                pass
        return mx

    def load_previous_checkpoints(self, last_batch):
        dfs = []
        for i in range(1, last_batch + 1):
            fp = os.path.join(self.checkpoint_dir, f"batch_{i:04d}.csv")
            if os.path.exists(fp):
                try:
                    dfs.append(pd.read_csv(fp, low_memory=False))
                except Exception:
                    pass
        return dfs

    # ---------------------------
    # Phase-2: Fetch metadata
    # ---------------------------
    def fetch_molecule_info(self, chembl_id):
        with self.cache_lock:
            if chembl_id in self.cache["molecule"]:
                return self.cache["molecule"][chembl_id]

        url = f"{self.base_url}/molecule/{chembl_id}.json"
        data = self.get_json(url)
        if not data:
            out = {"molecule_chembl_id": chembl_id}
        else:
            ms = data.get("molecule_structures") or {}
            out = {
                "molecule_chembl_id": chembl_id,
                "pref_name": data.get("pref_name", ""),
                "molecule_type": data.get("molecule_type", ""),
                "max_phase": data.get("max_phase", None),
                "canonical_smiles": ms.get("canonical_smiles", ""),
                "standard_inchi_key": ms.get("standard_inchi_key", ""),
            }

        with self.cache_lock:
            self.cache["molecule"][chembl_id] = out
        time.sleep(self.SOFT_SLEEP)
        return out

    def fetch_assay_info(self, assay_id):
        if not assay_id:
            return {"assay_chembl_id": assay_id}

        with self.cache_lock:
            if assay_id in self.cache["assay"]:
                return self.cache["assay"][assay_id]

        url = f"{self.base_url}/assay/{assay_id}.json"
        data = self.get_json(url)
        if not data:
            out = {"assay_chembl_id": assay_id}
        else:
            out = {
                "assay_chembl_id": assay_id,
                "assay_type": data.get("assay_type", ""),
                "assay_organism": data.get("assay_organism", ""),
                "assay_cell_type": data.get("assay_cell_type", ""),
                "confidence_score": data.get("confidence_score", None),
                "assay_description": str(data.get("description", ""))[:300],
                "bao_format_assay": data.get("bao_format", ""),
                "bao_label_assay": data.get("bao_label", ""),
            }

        with self.cache_lock:
            self.cache["assay"][assay_id] = out
        time.sleep(self.SOFT_SLEEP)
        return out

    def fetch_target_info(self, target_id):
        if not target_id:
            return {"target_chembl_id": target_id}

        with self.cache_lock:
            if target_id in self.cache["target"]:
                return self.cache["target"][target_id]

        url = f"{self.base_url}/target/{target_id}.json"
        data = self.get_json(url)
        out = {"target_chembl_id": target_id}

        if data:
            out.update({
                "target_pref_name_full": data.get("pref_name", ""),
                "target_type_full": data.get("target_type", ""),
                "target_organism_full": data.get("organism", ""),
            })

            # Attempt UniProt accession extraction (best-effort)
            acc = ""
            comps = data.get("target_components") or []
            for c in comps:
                xrefs = c.get("target_component_xrefs") or []
                for xr in xrefs:
                    src = str(xr.get("xref_src_db", "")).lower()
                    if src in ["uniprot", "uniprotkb"]:
                        acc = xr.get("xref_id", "") or acc
            out["uniprot_accession"] = acc

        with self.cache_lock:
            self.cache["target"][target_id] = out
        time.sleep(self.SOFT_SLEEP)
        return out

    def fetch_document_info(self, doc_id):
        if not doc_id:
            return {"document_chembl_id": doc_id}

        with self.cache_lock:
            if doc_id in self.cache["document"]:
                return self.cache["document"][doc_id]

        url = f"{self.base_url}/document/{doc_id}.json"
        data = self.get_json(url)
        if not data:
            out = {"document_chembl_id": doc_id}
        else:
            out = {
                "document_chembl_id": doc_id,
                "doi": data.get("doi", ""),
                "journal": data.get("journal", ""),
                "year": data.get("year", None),
                "pubmed_id": data.get("pubmed_id", ""),
            }

        with self.cache_lock:
            self.cache["document"][doc_id] = out
        time.sleep(self.SOFT_SLEEP)
        return out

    def parallel_fetch(self, ids, fn, desc):
        ids = [i for i in ids if isinstance(i, str) and i.strip() != ""]
        out = []
        if not ids:
            return pd.DataFrame()

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as ex:
            futs = [ex.submit(fn, _id) for _id in ids]
            for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
                try:
                    r = f.result()
                    if r:
                        out.append(r)
                except Exception:
                    continue

        return pd.DataFrame(out)

    def enrich_master(self, df):
        mol_ids = sorted(df["molecule_chembl_id"].dropna().unique().tolist())
        assay_ids = sorted(df["assay_chembl_id"].dropna().unique().tolist())
        target_ids = sorted(df["target_chembl_id"].dropna().unique().tolist())
        doc_ids = sorted(df["document_chembl_id"].dropna().unique().tolist())

        print("\n🔧 Fetching MOLECULE structures (SMILES/InChIKey)...")
        mol_df = self.parallel_fetch(mol_ids, self.fetch_molecule_info, "molecules")

        print("\n🔧 Fetching ASSAY metadata...")
        assay_df = self.parallel_fetch(assay_ids, self.fetch_assay_info, "assays")

        print("\n🔧 Fetching TARGET metadata (UniProt best-effort)...")
        target_df = self.parallel_fetch(target_ids, self.fetch_target_info, "targets")

        print("\n🔧 Fetching DOCUMENT metadata (DOI/journal/year)...")
        doc_df = self.parallel_fetch(doc_ids, self.fetch_document_info, "documents")

        out = df.merge(mol_df, on="molecule_chembl_id", how="left")
        out = out.merge(assay_df, on="assay_chembl_id", how="left")
        out = out.merge(target_df, on="target_chembl_id", how="left")
        out = out.merge(doc_df, on="document_chembl_id", how="left")
        return out

    # ---------------------------
    # Optional chemistry descriptors
    # ---------------------------
    def add_rdkit_chemistry(self, df):
        if not USE_RDKIT:
            print("⚠️ RDKit not available. Skipping chemistry descriptors.")
            return df

        def calc(smiles):
            if not isinstance(smiles, str) or not smiles:
                return (None,) * 10
            m = Chem.MolFromSmiles(smiles)
            if not m:
                return (None,) * 10
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            inchi = Chem.MolToInchiKey(m)
            return (
                float(Descriptors.MolWt(m)),
                float(Descriptors.MolLogP(m)),
                float(Descriptors.TPSA(m)),
                int(Descriptors.NumHDonors(m)),
                int(Descriptors.NumHAcceptors(m)),
                int(Descriptors.NumRotatableBonds(m)),
                int(Descriptors.RingCount(m)),
                int(m.GetNumHeavyAtoms()),
                scaf,
                inchi
            )

        cols = ["MolWt", "cLogP_RDKit", "TPSA", "HBD", "HBA", "RotB", "RingCount",
                "HeavyAtoms", "MurckoScaffold", "RDKit_InchiKey"]

        tmp = df["canonical_smiles"].fillna("").apply(calc).apply(pd.Series)
        tmp.columns = cols
        return pd.concat([df, tmp], axis=1)

    # ---------------------------
    # Summary / JSON type conversion
    # ---------------------------
    def convert_numpy_types(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_numpy_types(x) for x in obj]
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                return obj
        return obj

    def generate_summary(self, df, total_time_sec):
        out = {
            "processing_info": {
                "total_molecules_processed": int(self.stats["total_processed_molecules"]),
                "successful_molecules": int(self.stats["successful_molecules"]),
                "failed_molecules": int(self.stats["failed_molecules"]),
                "success_rate": (
                    f"{(self.stats['successful_molecules'] / max(1, self.stats['total_processed_molecules'])) * 100:.1f}%"
                ),
                "total_time_minutes": float(total_time_sec / 60.0),
                "end_time": datetime.now().isoformat(timespec="seconds"),
            },
            "data_summary": {
                "total_rows": int(len(df)),
                "unique_molecules": int(df["molecule_chembl_id"].nunique()),
                "unique_targets": int(df["target_chembl_id"].nunique()) if "target_chembl_id" in df.columns else 0,
                "rows_with_smiles": int((df.get("canonical_smiles", pd.Series([])).fillna("") != "").sum()),
                "rows_with_inchikey": int((df.get("standard_inchi_key", pd.Series([])).fillna("") != "").sum()),
                "rows_with_doi": int((df.get("doi", pd.Series([])).fillna("") != "").sum()),
                "pchembl_nonnull": int(df.get("pchembl_value", pd.Series([])).notna().sum()),
            }
        }
        return out

    # ---------------------------
    # Batch processing
    # ---------------------------
    def process_batch(self, batch_ids, batch_num, total_batches):
        all_acts = []
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_id = {executor.submit(self.fetch_molecule_activities, cid): cid for cid in batch_ids}

            for future in tqdm(as_completed(future_to_id),
                               total=len(batch_ids),
                               desc=f"Batch {batch_num}/{total_batches}",
                               unit="mol",
                               leave=False):
                cid = future_to_id[future]
                try:
                    acts, st = future.result()
                    self.stats["total_processed_molecules"] += 1
                    if st.get("status") == "success" and acts:
                        self.stats["successful_molecules"] += 1
                        self.stats["total_activities"] += len(acts)
                        all_acts.extend(acts)
                    else:
                        self.stats["failed_molecules"] += 1
                except Exception:
                    self.stats["total_processed_molecules"] += 1
                    self.stats["failed_molecules"] += 1

        if all_acts:
            dfb = pd.DataFrame(all_acts)
            checkpoint_file = os.path.join(self.checkpoint_dir, f"batch_{batch_num:04d}.csv")
            dfb.to_csv(checkpoint_file, index=False, encoding="utf-8")
            return dfb
        return pd.DataFrame()

    def run(self, all_ids):
        total_ids = len(all_ids)
        total_batches = math.ceil(total_ids / self.BATCH_SIZE)

        print("=" * 72)
        print("ChEMBL Coumarin Retrieval and Metadata Assembly Script")
        print("=" * 72)
        print(f"• Total IDs: {total_ids:,}")
        print(f"• Batch size: {self.BATCH_SIZE}")
        print(f"• Workers: {self.MAX_WORKERS}")
        print(f"• RDKit descriptors: {'ON' if USE_RDKIT else 'OFF'}")
        print("=" * 72)

        start_time = time.time()

        last_cp = self.find_last_checkpoint()
        if last_cp > 0:
            print(f"⏭️  Found checkpoint up to batch {last_cp}. Resuming...")
            all_dfs = self.load_previous_checkpoints(last_cp)
            start_batch = last_cp + 1
        else:
            all_dfs = []
            start_batch = 1

        for b in range(start_batch, total_batches + 1):
            s = (b - 1) * self.BATCH_SIZE
            e = min(b * self.BATCH_SIZE, total_ids)
            batch_ids = all_ids[s:e]

            dfb = self.process_batch(batch_ids, b, total_batches)
            if not dfb.empty:
                all_dfs.append(dfb)

            done = min(b * self.BATCH_SIZE, total_ids)
            pct = (done / total_ids) * 100
            elapsed = time.time() - start_time
            print(f"\n📈 Progress: {done:,}/{total_ids:,} ({pct:.1f}%) | elapsed {elapsed/60:.1f} min")
            print(f"   Activities so far: {self.stats['total_activities']:,} | success mol: {self.stats['successful_molecules']:,}")

        if not all_dfs:
            print("❌ No data collected.")
            return pd.DataFrame()

        print("\n🔗 Concatenating batches...")
        df = pd.concat(all_dfs, ignore_index=True)

        print("\n🧩 Enriching with Molecule/Assay/Target/Document metadata...")
        df = self.enrich_master(df)

        if USE_RDKIT:
            print("\n🧪 Adding RDKit chemistry descriptors + Murcko scaffold...")
            df = self.add_rdkit_chemistry(df)

        # Save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(self.output_dir, f"coumarin_master_{ts}.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8")

        # Summary
        summary = self.generate_summary(df, time.time() - start_time)
        summary = self.convert_numpy_types(summary)
        out_json = os.path.join(self.output_dir, f"summary_{ts}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 72)
        print("✅ DONE")
        print("=" * 72)
        print(f"• Master CSV: {out_csv}")
        print(f"• Summary:    {out_json}")
        print(f"• Rows:       {len(df):,}")
        print(f"• Unique mol: {df['molecule_chembl_id'].nunique():,}")
        if "canonical_smiles" in df.columns:
            print(f"• SMILES coverage: {(df['canonical_smiles'].fillna('')!='').mean()*100:.1f}%")
        if "doi" in df.columns:
            print(f"• DOI coverage: {(df['doi'].fillna('')!='').mean()*100:.1f}%")
        print("=" * 72)

        return df


def main():
    ids_file = "ChEMBL_ID.txt"
    if not os.path.exists(ids_file):
        print("❌ ChEMBL_ID.txt not found in current directory.")
        return

    with open(ids_file, "r", encoding="utf-8") as f:
        all_ids = [ln.strip() for ln in f if ln.strip().startswith("CHEMBL")]

    if not all_ids:
        print("❌ No valid CHEMBL IDs found in ChEMBL_ID.txt.")
        return

    scanner = ChEMBLMasterScanner()
    try:
        scanner.run(all_ids)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. You can resume later (checkpoints kept).")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
