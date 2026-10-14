from helpers import Papers, levels, loadPapers, loadSequences, \
        deblur, classify, paperMerge, collapse, fragmentInsert, makeOutputDir
from sys import argv, exit



def main():

    if len(argv) != 6:
        exit("Usage: python3  denoise.py  paper_metadata  taxonomic_level  classifier_model  reference_database  output_dir")

    if argv[2].lower() not in levels:
        exit("Choose a valid taxonomic level (e.g. Kingdom, Phylum, Class, Order, Family, Genus and Species).")

    papers = loadPapers(argv[1])

    for paper in papers:
        outputDir = makeOutputDir(paper, argv[5])
        loadSequences(paper)
        deblur(paper, outputDir)
        classify(paper, argv[3], outputDir)
    
    mergedPaper = Papers("merged", "merged", "merged")
    outputDir = makeOutputDir(mergedPaper, argv[5])

    paperMerge(mergedPaper, papers, outputDir)
    collapse(mergedPaper, argv[2], outputDir)
    fragmentInsert(mergedPaper, argv[4], outputDir)



if __name__ == "__main__":
    main()