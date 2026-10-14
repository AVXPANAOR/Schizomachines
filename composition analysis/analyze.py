from helpers import Papers, levels,\
    phylogeneticAnalysis, alphaRarefy, differentialAbundance, collapse,\
    loadData, loadMetadata, makeOutputDir
from sys import argv, exit



def main():

    if len(argv) != 9:
        exit("Usage: python3  analyze.py  merged_tree  filtered_table  taxonomy  taxonomic_level  sampling_depth  max_depth  sample_metadata  output_dir")

    if not (argv[5].isdigit() and argv[6].isdigit()):
        exit("Choose a positive integer as sampling depth.")

    if argv[4].lower() not in levels:
        exit("Choose a valid taxonomic level (e.g. Kingdom, Phylum, Class, Order, Family, Genus and Species).")

    
    mergedPaper = Papers("merged", "merged", "merged")
    outputDir = makeOutputDir(mergedPaper, argv[8])
    sampleMetadata = loadMetadata(argv[7])

    loadData(mergedPaper, argv[1], argv[2], argv[3])
    if int(argv[5]) == 0:
        alphaRarefy(mergedPaper, sampleMetadata, int(argv[6]), outputDir)
    else:
        phylogeneticAnalysis(mergedPaper, sampleMetadata, int(argv[5]), outputDir)
        collapse(mergedPaper, argv[4], outputDir)
        differentialAbundance(mergedPaper, sampleMetadata, outputDir)
    

if __name__ == "__main__":
    main()