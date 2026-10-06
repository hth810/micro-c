"""Public metadata identity and paired/unpaired file interpretation."""
import pytest

from scripts.audit_raw_read_availability_followup import parse_ena, parse_geo


def ena_text():
    return ("run_accession\texperiment_accession\tstudy_accession\tinstrument_model\tlibrary_layout\tread_count\tbase_count\tfastq_ftp\tfastq_bytes\tfastq_md5\n"
            "SRR1\tSRX1\tPRJ1\tNextSeq 2000\tPAIRED\t10\t2020\t"
            "ftp.sra.ebi.ac.uk/x/SRR1.fastq.gz;ftp.sra.ebi.ac.uk/x/SRR1_1.fastq.gz;ftp.sra.ebi.ac.uk/x/SRR1_2.fastq.gz\t"
            "11;22;33\t" + ";".join(["a" * 32, "b" * 32, "c" * 32]) + "\n")


def test_geo_preserves_repeated_relations_and_characteristics():
    data = parse_geo("!Sample_geo_accession = GSM1\n!Sample_relation = SRA: SRX1\n!Sample_relation = BioSample: A\n!Sample_characteristics_ch1 = genotype: WT\n")
    assert data["Sample_geo_accession"] == ["GSM1"]
    assert len(data["Sample_relation"]) == 2


def test_fastq_manifest_distinguishes_unpaired_and_pair_bytes():
    row = parse_ena(ena_text(), "SRR1")
    assert [f["role"] for f in row["fastq_files"]] == ["unpaired", "R1", "R2"]
    assert row["paired_fastq_gzip_bytes"] == 55
    assert row["fastq_files"][0]["https_url"].startswith("https://ftp.sra.ebi.ac.uk/")


def test_wrong_run_identity_is_rejected():
    with pytest.raises(ValueError, match="identity"):
        parse_ena(ena_text(), "SRR2")
