from helpers import loadPapers, loadMetadata, loadSequences,\
    demultiplex, trimAdapters, merge, qualityFilter, makeOutputDir
from sys import argv, exit
from gc import collect



def main():

    if len(argv) != 4:
        exit("Usage: python3  process.py  paper_metadata  sample_metadata  output_dir")

    papers = loadPapers(argv[1])
    sampleMetadata = loadMetadata(argv[2])

    for paper in papers:
        outputDir = makeOutputDir(paper, argv[3])
        loadSequences(paper)
        demultiplex(paper, sampleMetadata, outputDir)
        trimAdapters(paper, outputDir)
        merge(paper, outputDir)
        qualityFilter(paper, outputDir)

        del paper
        collect()


if __name__ == "__main__":
    main()