import qiime2
from qiime2.plugins import quality_filter, vsearch, demux, metadata, feature_table, taxa
from qiime2.plugins.cutadapt.methods import demux_single, demux_paired, trim_paired, trim_single
from qiime2.plugins.deblur.methods import denoise_16S
from qiime2.plugins.deblur.visualizers import visualize_stats
from qiime2.plugins.feature_classifier.actions import classify_sklearn
from qiime2.plugins.fragment_insertion.methods import sepp, filter_features
from qiime2.plugins.diversity.actions import core_metrics_phylogenetic, alpha_rarefaction, alpha_group_significance, beta_group_significance
from qiime2.plugins.composition.actions import ancombc, da_barplot
from csv import DictReader, DictWriter
from re import search
from multiprocessing import cpu_count
from os import path, makedirs
import pandas as pd
import numpy as np



class Papers:
    def __init__(self, name, layout, platform,
                 demuxed=None, sequencePath=None,
                 fprimer=None, rprimer=None, 
                 sequences=None, table=None, taxonomy=None, tree=None,
                 metrics=None,
                 ltrim=None, rtrim=None):
        
        self.name = name
        self.layout = layout
        self.platform = platform
        self.demuxed = bool(demuxed)
        self.sequencePath = sequencePath
        self.primers = [fprimer, rprimer]
        self.sequences = sequences
        self.table = table
        self.taxonomy = taxonomy
        self.tree = tree
        self.metrics = metrics
        self.ltrim = ltrim
        self.rtrim = rtrim


levels = {"kingdom":1, "phylum":2, "class":3, "order":4, "family":5, "genus":6, "species":7}


def loadPapers(paperMetadata):

    papers = []
    with open(paperMetadata) as file:
        reader = DictReader(file)
        for row in reader:
            paper = Papers(row["name"], row["layout"], row["platform"],
                           row["demuxed"], row["sequencePath"],
                           row["fprimer"], row["rprimer"], 
                           ltrim=int(row["ltrim"]), rtrim=int(row["rtrim"]))
            papers.append(paper)
    return papers


def loadMetadata(sampleMetadata):
    return qiime2.Metadata.load(sampleMetadata)
    

def makeOutputDir(paper, rootDir):

    outputDir = path.join(rootDir, paper.name)
    if not path.isdir(outputDir):
        makedirs(outputDir)

    return outputDir


def loadSequences(paper):

    if search(".qza$", paper.sequencePath):
        paper.sequences = qiime2.Artifact.load(paper.sequencePath)

    else:
        if paper.demuxed:
            format = f"SampleData[{'PairedEnd' if paper.layout == 'paired' else ''}SequencesWithQuality]"
            type = f"{paper.layout.capitalize()}EndFastqManifestPhred33V2"
        else:
            format = f"Multiplexed{paper.layout.capitalize()}EndBarcodeInSequence"
            type = f"Multiplexed{paper.layout.capitalize()}EndBarcodeInSequenceDirFmt"

        paper.sequences = qiime2.Artifact.import_data(format, paper.sequencePath, view_type=type)


def loadData(paper, treePath=None, tablePath=None, taxonomyPath=None):
    if treePath:
        paper.tree = qiime2.Artifact.load(treePath)
    if tablePath:
        paper.table = qiime2.Artifact.load(tablePath)
    if taxonomyPath:
        paper.taxonomy = qiime2.Artifact.load(taxonomyPath)


def demultiplex(paper, sampleMetadata, outputDir):

    if not paper.demuxed:
        barcodes = sampleMetadata.get_column('Barcode')
        if paper.layout == "paired":
            paper.sequences = demux_paired(seqs=paper.sequences, forward_barcodes=barcodes,
                                           cores=cpu_count()).per_sample_sequences
        else:
            paper.sequences = demux_single(seqs=paper.sequences, barcodes=barcodes,
                                           cores=cpu_count()).per_sample_sequences

    demuxSummary = demux.visualizers.summarize(paper.sequences)
    demuxSummary.visualization.save(path.join(outputDir, f"{paper.name}-demuxed.qzv"))


def trimAdapters(paper, outputDir):

    illuminaUniversal = ["AGATCGGAAGAG"]

    if paper.platform == "illumina":
        adapter = illuminaUniversal
    else:
        adapter = reverseComplement(paper.primers)

    if paper.layout == "paired":
        paper.sequences = trim_paired(demultiplexed_sequences=paper.sequences,
                                      front_f=paper.primers[:1],
                                      front_r=paper.primers[1:],
                                      match_adapter_wildcards=True, cores=cpu_count()).trimmed_sequences
        
        paper.sequences = trim_paired(demultiplexed_sequences=paper.sequences,
                                      adapter_f=adapter,
                                      adapter_r=adapter,
                                      match_adapter_wildcards=True, cores=cpu_count()).trimmed_sequences
    else: 
        paper.sequences = trim_single(demultiplexed_sequences=paper.sequences,
                                      front=paper.primers,
                                      match_adapter_wildcards=True, cores=cpu_count()).trimmed_sequences
        
        paper.sequences = trim_single(demultiplexed_sequences=paper.sequences,
                                      adapter=adapter,
                                      match_adapter_wildcards=True, cores=cpu_count()).trimmed_sequences

    trimSummary = demux.visualizers.summarize(paper.sequences)
    trimSummary.visualization.save(path.join(outputDir, f"{paper.name}-trimmed.qzv"))


def merge(paper, outputDir):

    if paper.layout == "paired":
        paper.sequences = vsearch.methods.merge_pairs(paper.sequences, threads=0).merged_sequences
        mergedSummary = demux.visualizers.summarize(paper.sequences)
        mergedSummary.visualization.save(path.join(outputDir, f"{paper.name}-merged.qzv"))
    else:
        pass


def qualityFilter(paper, outputDir):

    filtered = quality_filter.methods.q_score(paper.sequences)
    filtered.filtered_sequences.save(path.join(outputDir, f"{paper.name}-filtered.qza"))

    filterStats = metadata.visualizers.tabulate(filtered.filter_stats.view(qiime2.Metadata))
    filterStats.visualization.save(path.join(outputDir, f"{paper.name}-filter-stats.qzv"))

    paper.sequences = filtered.filtered_sequences


def deblur(paper, outputDir):
    
    deblured = denoise_16S(paper.sequences,
                           trim_length=paper.rtrim, left_trim_len=paper.ltrim,
                           sample_stats=True, jobs_to_start=cpu_count())
    paper.sequences = deblured.representative_sequences
    paper.sequences.save(path.join(outputDir, f"{paper.name}-deblurred-seqs.qza"))

    paper.table = deblured.table
    paper.table.save(path.join(outputDir, f"{paper.name}-feature-table.qza"))

    deblurStats = visualize_stats(deblured.stats)
    deblurStats.visualization.save(path.join(outputDir, f"{paper.name}-deblur-stats.qzv"))

    featureTableSummary = feature_table.visualizers.summarize(paper.table)
    featureTableSummary.visualization.save(path.join(outputDir, f"{paper.name}-feature-summary.qzv"))

    featureTableTabulate = feature_table.visualizers.tabulate_seqs(paper.sequences)
    featureTableTabulate.visualization.save(path.join(outputDir, f"{paper.name}-deblurred-seqs.qzv"))


def classify(paper, classifier, outputDir):
    
    model = qiime2.Artifact.load(classifier)
    taxonomy = classify_sklearn(classifier=model, reads=paper.sequences, n_jobs=-1)
    paper.taxonomy = taxonomy.classification
    paper.taxonomy.save(path.join(outputDir, f"{paper.name}-taxonomy.qza"))

    taxonomyTable = metadata.visualizers.tabulate(paper.taxonomy.view(qiime2.Metadata))
    taxonomyTable.visualization.save(path.join(outputDir, f"{paper.name}-taxonomy.qzv"))
    

def paperMerge(mergedPaper, papers, outputDir):
    
    mergedPaper.sequences = feature_table.methods.merge_seqs([paper.sequences for paper in papers]).merged_data
    mergedPaper.sequences.save(path.join(outputDir, f"{mergedPaper.name}-deblurred-seqs.qza"))
    featureTableTabulate = feature_table.visualizers.tabulate_seqs(mergedPaper.sequences)
    featureTableTabulate.visualization.save(path.join(outputDir, f"{mergedPaper.name}-deblurred-seqs.qzv"))

    mergedPaper.table = feature_table.methods.merge([paper.table for paper in papers], 
                                                    overlap_method="error_on_overlapping_sample").merged_table
    mergedPaper.table.save(path.join(outputDir, f"{mergedPaper.name}-feature-table.qza"))
    featureTableSummary = feature_table.visualizers.summarize(mergedPaper.table)
    featureTableSummary.visualization.save(path.join(outputDir, f"{mergedPaper.name}-feature-summary.qzv"))
    
    mergedPaper.taxonomy = feature_table.methods.merge_taxa([paper.taxonomy for paper in papers]).merged_data
    mergedPaper.taxonomy.save(path.join(outputDir, f"{mergedPaper.name}-taxonomy.qza"))
    taxonomyTable = metadata.visualizers.tabulate(mergedPaper.taxonomy.view(qiime2.Metadata))
    taxonomyTable.visualization.save(path.join(outputDir, f"{mergedPaper.name}-taxonomy.qzv"))


def fragmentInsert(paper, SEPPreferenceDatabase, outputDir):

    referenceDatabase = qiime2.Artifact.load(SEPPreferenceDatabase)
    seppResults = sepp(representative_sequences=paper.sequences,
                       reference_database=referenceDatabase,
                       threads=cpu_count())
    paper.tree = seppResults.tree
    paper.tree.save(path.join(outputDir, f"{paper.name}-SEPP-tree.qza"))
    seppResults.placements.save(path.join(outputDir, f"{paper.name}-SEPP-placements.qza"))

    filteredFeatures = filter_features(table=paper.table, tree=paper.tree)
    paper.table = filteredFeatures.filtered_table
    paper.table.save(path.join(outputDir, f"{paper.name}-filtered-table.qza"))
    filteredFeatures.removed_table.save(path.join(outputDir, f"{paper.name}-removed-table.qza"))


def collapse(paper, level, outputDir):
    
    paper.table = taxa.methods.collapse(table=paper.table,
                                      taxonomy=paper.taxonomy,
                                      level=levels[level.lower()]).collapsed_table
    paper.table.save(path.join(outputDir, f"{paper.name}-collapsed.qza"))


def alphaRarefy(paper, sampleMetadata, maxSamplingDepth, outputDir):

    alphaRarefaction = alpha_rarefaction(table=paper.table,
                                              phylogeny=paper.tree,
                                              max_depth=maxSamplingDepth,
                                              metadata=sampleMetadata)

    alphaRarefaction.visualization.save(path.join(outputDir, f"{paper.name}-alpha-rarefaction.qzv"))


def phylogeneticAnalysis(paper, sampleMetadata, samplingDepth, outputDir):

    paper.metrics = core_metrics_phylogenetic(
        phylogeny=paper.tree,
        table=paper.table,
        sampling_depth=samplingDepth,
        metadata=sampleMetadata,
        n_jobs_or_threads='auto'
    )

    artifact_outputs = [
        "rarefied_table",
        "faith_pd_vector",
        "observed_features_vector",
        "shannon_vector",
        "evenness_vector",
        "unweighted_unifrac_distance_matrix",
        "weighted_unifrac_distance_matrix",
        "jaccard_distance_matrix",
        "bray_curtis_distance_matrix",
        "unweighted_unifrac_pcoa_results",
        "weighted_unifrac_pcoa_results",
        "jaccard_pcoa_results",
        "bray_curtis_pcoa_results",
    ]

    visualization_outputs = [
        "unweighted_unifrac_emperor",
        "weighted_unifrac_emperor",
        "jaccard_emperor",
        "bray_curtis_emperor",
    ]

    # Save QIIME artifacts.
    for metric in artifact_outputs:
        result = getattr(paper.metrics, metric)

        result.save(
            path.join(
                outputDir,
                f"{paper.name}-{metric}.qza"
            )
        )

        # Alpha-diversity group significance.
        if metric.endswith("_vector"):
            alphaGroupSignificance = alpha_group_significance(
                alpha_diversity=result,
                metadata=sampleMetadata
            )

            alphaGroupSignificance.visualization.save(
                path.join(
                    outputDir,
                    f"{paper.name}-{metric}-group-significance.qzv"
                )
            )

        # Beta-diversity group significance.
        elif metric.endswith("_distance_matrix"):
            for method in ["permanova", "permdisp", "anosim"]:

                betaGroupSignificance = beta_group_significance(
                    distance_matrix=result,
                    metadata=sampleMetadata.get_column("Disease"),
                    pairwise=True,
                    method=method
                )

                betaGroupSignificance.visualization.save(
                    path.join(
                        outputDir,
                        f"{paper.name}-{metric}-{method}.qzv"
                    )
                )

    # Save Emperor visualizations.
    for metric in visualization_outputs:
        result = getattr(paper.metrics, metric)

        result.save(
            path.join(
                outputDir,
                f"{paper.name}-{metric}.qzv"
            )
        )

def differentialAbundance(paper, sampleMetadata, outputDir):
    
    for disease in ["BD", "MDD", "SCZ", "Healthy"]:
        ancombcDisease = ancombc(table=paper.table,
                                metadata=sampleMetadata,
                                formula='Disease',
                                reference_levels=f'Disease::{disease}',
                                p_adj_method='BH',
                                alpha=0.001).differentials
        ancombcDisease.save(path.join(outputDir, f"{paper.name}-ref{disease}-ancombc.qza"))

        daBarplot = da_barplot(data=ancombcDisease, significance_threshold=0.001)
        daBarplot.visualization.save(path.join(outputDir, f"{paper.name}-ref{disease}-da-barplot.qzv"))


def filterTable(paper, sampleMetadata, disease, outputDir):

    outputSub = path.join(outputDir, disease)

    filteredTable = feature_table.methods.filter_samples(paper.table, metadata=sampleMetadata, \
                                                         where=f'[Disease]={disease}').filtered_table
    filteredTable.save(path.join(outputSub, f"{disease}-filtered.qza"))
    
    df = filteredTable.view(pd.DataFrame)
    df = df.drop(df[df.mean(axis=1, numeric_only=True) < 2].index)
    cTable = df.to_dict('records')

    clearedTable = []

    for sample in cTable:
        cOTU = sample["#OTU ID"].split(";")[-2:]
        family = cOTU[0].lstrip("f__")
        genus = cOTU[1].lstrip("g__")
        if (cOTU[0] not in ["", "uncultured"]):
            cSample = sample
            cSample["#OTU ID"] = family + (". " if (genus != "") else "") + genus
            clearedTable.append(cSample)
    
    outputSub = path.join(outputDir, disease)
    if not path.isdir(outputSub):
        makedirs(outputSub)

    tsvPath = path.join(outputSub, f"cleared-{disease}.tsv")
    with open(tsvPath, "w") as tsvfile:
        writer = DictWriter(tsvfile, fieldnames=list(clearedTable[0].keys()) ,delimiter="\t")
        writer.writeheader()
        writer.writerows(clearedTable)

    return tsvPath

def reverseComplement(primers):
    wildcards = ["AT", "CG", "RY", "SS", "WW", "KM", "BV", "DH", "NN"]
    reverseComplements = []

    for primer in primers:
        complement = ""
        for base in primer:
            for wildcard in wildcards:
                if base in wildcard:
                    complement += wildcard[1-wildcard.index(base)]
        reverseComplements.append(complement[::-1])
    return reverseComplements
