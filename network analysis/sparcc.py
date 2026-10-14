import subprocess
import shlex
from os import makedirs, path
import pandas as pd
import numpy as np
from itertools import combinations
from multiprocessing import cpu_count


N_PERMUTATIONS = 1000
INPUT_DIR = "raw"
OUTPUT_BASE_DIR = "permutations"
INPUT_FILES = {
    "BD": path.join(INPUT_DIR, "BD.tsv"),
    "MDD": path.join(INPUT_DIR, "MDD.tsv"),
    "SCZ": path.join(INPUT_DIR, "SCZ.tsv"),
    "Healthy": path.join(INPUT_DIR, "Healthy.tsv")
}


def main():
    
    combs = combinations(INPUT_FILES, 2)
    for (disease1, disease2) in combs:
        
        df1 = pd.read_csv(INPUT_FILES[disease1], header=0, index_col=0, sep="\t").T
        df2 = pd.read_csv(INPUT_FILES[disease2], header=0, index_col=0, sep="\t").T
        cols = df1.columns.intersection(df2.columns)
        cdf = pd.concat([df1[cols], df2[cols]], ignore_index=True)
        conc = disease1 + disease2
   
        print("")
        print(f"Calculating {conc} Correlations")

        outputDir = path.join(OUTPUT_BASE_DIR, conc)
        if not path.isdir(outputDir):
            makedirs(outputDir)
        
        countsDir = path.join(outputDir, "counts")
        if not path.isdir(countsDir):
            makedirs(countsDir)

        corDir = path.join(outputDir, "cor")
        if not path.isdir(corDir):
            makedirs(corDir)

        covDir = path.join(outputDir, "cov")
        if not path.isdir(covDir):
            makedirs(covDir)

        for i in range(N_PERMUTATIONS):
            if (i + 1) % max(1, (N_PERMUTATIONS // 100)) == 0 or (i + 1) == N_PERMUTATIONS:
                progress = (i + 1) / N_PERMUTATIONS
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f"\r  Progress: |{bar}| {progress:.1%} ({i+1}/{N_PERMUTATIONS})", end='')

            cdf = cdf.sample(frac=1, ignore_index=False)
         
            pdf1 = cdf.iloc[:len(df1)].T
            pdf2 = cdf.iloc[len(df1):].T
    
            fileName1 = f"{i}_1.tsv"
            fileName2 = f"{i}_2.tsv"

            for (pdf, fileName) in [(pdf1, fileName1), (pdf2, fileName2)]:

                tsvPath = path.join(countsDir, fileName)
                corPath = path.join(corDir, fileName)
                covPath = path.join(covDir, fileName)
                pdf.to_csv(tsvPath, sep="\t")
                
                cmd = shlex.split(f'fastspar -c {tsvPath} -r {corPath} -a {covPath} -t {cpu_count()} -y')  
                process = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                
if __name__ == "__main__":
    main()
