library('estimatr')
library(emmeans)


BD <- read.csv("shannon/BD.tsv", sep="\t", header = TRUE, row.names = 1)
MDD <- read.csv("shannon/MDD.tsv", sep="\t", header = TRUE, row.names = 1)
SCZ <- read.csv("shannon/SCZ.tsv", sep="\t", header = TRUE, row.names = 1)
meta <- read.csv("metadata.tsv", sep="\t", header = TRUE, row.names = 1)

BDdata <- merge(BD, meta, by='row.names')[-c(29, 62, 124),]
model <- lm_robust(Y ~ Study + Disease, data = BDdata, se_type="HC3")
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")

MDDdata <- merge(MDD, meta, by='row.names')[-c(42, 92, 465, 449, 490),]
model <- lm_robust(Y ~ Study + Disease, data = MDDdata, se_type="HC3")
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")

SCZdata <- merge(SCZ, meta, by='row.names')[-c(219, 269, 287, 337, 348),]
model <- lm_robust(Y ~ Study + Disease, data = SCZdata, se_type="HC3")
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")

