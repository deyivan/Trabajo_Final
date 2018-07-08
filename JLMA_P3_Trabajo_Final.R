###############################################################
##    T r a b a j o   F i n a l   D S I 
###############################################################
##    C l u s t e r i n g   +   P r e d i c c i ó n
###############################################################
##    A u t o r :   J o s é   L u i s   M u ñ o z   A m p u e r o
##    D e s a r r o l l o   d e   S i s t.   I n t e l i g e n t e s
##    M á s t e r   e n   I n g e n i e r í a   I n f o r m á t i c a
##    ( E S I - C i u d a d   R e a l ,   U C L M )
###############################################################

rm(list = ls()) # borramos todos los objetos

##############################################################
##    C a r g a   d e   l i b r e r í a s   e x t e r n a s
##############################################################

needed.packages <- c("Amelia", "rgl", "Rtsne", "pscl", "Metrics")
new.packages <- needed.packages[!(needed.packages %in% installed.packages()[,"Package"])]

library(Amelia)     # Valores perdidos
library(rgl)        # Plot 3D
library(Rtsne)      # t-SNE (implementación Barnes-Hut)
library(pscl)       # Estadístico R2-McFadden
library(Metrics)    # Métrica RMSE

##############################################################
##   C l u s t e r i n g   c o m o   a p  o y o   a   l a     
##  S e l e c c  i ó n   d e   c  a r á c t e r  í s t i c a s
##############################################################

# Carga del fichero maestro para extraer características de los datos
MasterData <- read.table(file="Development Data Sets/algebra_2005_2006_master_preprocessed.txt", 
                         row.names = NULL, sep = "\t", header = TRUE, fill = TRUE, na.strings = "", blank.lines.skip = TRUE)[
                           c("Anon.Student.Id", "Problem.Hierarchy", "Problem.Name", "Step.Name",
                             "Step.Start.Time", "First.Transaction.Time", "Correct.Transaction.Time", "Step.End.Time",
                             "Problem.View", "Correct.First.Attempt", "Incorrects",	"Hints",	"Corrects",	
                             "Step.Duration..sec.", "Correct.Step.Duration..sec.", "Error.Step.Duration..sec.",	
                             "KC.Default.",	"KC.Additional_1.",	"KC.Additional_2.",	"KC.Additional_3."
                             )
                           ]
# Gráfico de valores perdidos vs observados
missmap(MasterData, main = "Missing values vs observed", legend = FALSE, y.labels = c(""), y.at = c(1), margins = c(9.5,0))
Master_Set <- data.matrix(MasterData) # Convertimos el dataset matriz de números
Master_Set [is.na(Master_Set)] <- -1  # Reemplazamos NA por -1

# Reduce la dimensionalidad de la muestra a 3D mediante TSNE (implementación Barnes-Hut)
tsne <- Rtsne(Master_Set, check_duplicates = FALSE, pca = FALSE, perplexity=50, theta=0.5, dims=3) 

# Mostrar resultados t-SNE en 2D
plot(tsne$Y, pch = 16)

# Mostrar resultados t-SNE en 3D
plot3d(tsne$Y, pch = 16)

# Preparar variables auxiliares para plot "ampliado"
cols <- rainbow(2)  # Dos colores
L <- list("Wrong first attempt", "Correct first attempt") # Dos categorías

# Mostrar resultados t-SNE en 2D en términos de la variable dependiente "Correct.First.Attempt"
plot(tsne$Y, pch = 16, col = cols[Master_Set[,"Correct.First.Attempt"] + 1])
legend("topright", legend = L, pch = 16, col= cols)

# Mostrar resultados t-SNE en 3D en términos de la variable dependiente "Correct.First.Attempt"
plot3d(tsne$Y, col = cols[Master_Set[,"Correct.First.Attempt"] + 1])
legend3d("topright", legend = L, pch = 16, col = cols )

##############################################################
##    C a r g a   d e l   C o n j u n t o   d e   D a t o s   
##            d e   E n t r e n a m i e n t o
##############################################################

# Carga del fichero de train en base a las características seleccionadas
TrainData.cs <- read.table(file="Development Data Sets/algebra_2005_2006_train_preprocessed.txt", 
                        row.names = NULL, sep = "\t", header = TRUE, fill = TRUE, na.strings = "", blank.lines.skip = TRUE)[
                          c("Problem.View",
                            "Hints",
                            "Correct.First.Attempt", 
                            "Incorrects",
                            "Error.Step.Duration..sec.",
                            "KC.Default.",	"KC.Additional_1.",	"KC.Additional_2.",	"KC.Additional_3."
                            )
                          ]
Training_Set <- data.matrix(TrainData.cs) # Convertimos el dataset matriz de números
Training_Set[is.na(Training_Set)] <- -1   # Reemplazamos NA por -1

##############################################################
##  A j u s t e   y   e v a l u c i ó n   d e l   m o d e l o
##############################################################

# Ajustamos el modelo mediante una MLG, particularizando un modelo de regresión logístico binomial 
model <- glm(Correct.First.Attempt ~ 
               Problem.View + 
               Hints + 
               Incorrects + 
               Error.Step.Duration..sec. +
               KC.Default. + KC.Additional_1. + KC.Additional_2. +	KC.Additional_3.,
             family = binomial(link = 'logit'), maxit = 100,
             data = as.data.frame(Training_Set))
summary(model)  # Resumen del ajuste del modelo
anova(model)    # Análisis de la varianza del modelo
pR2(model)["McFadden"]  # Extracción del estadístico pseudo-R2-McFadden para modelos de regresión logística

##############################################################
##    C a r g a   d e l   C o n j u n t o   d e   D a t o s   
##                      d e   T e s t
##############################################################

# Se reutilizará el fichero maestro eliminando, precisamente, la variable dependiente que se pretende clasificar
Testing_Set <- as.data.frame(subset(Master_Set, select = -c(Correct.First.Attempt)))

##############################################################
##  P r e d i c c i ó n   y   c á l c u l o   d e l   e r r o r
##############################################################

# Extracción de los valores reales observados
actual <- Master_Set[, c("Correct.First.Attempt")]
# Extracción de los valores estimados
predicted <- predict(model, newdata = Testing_Set, type = 'response')
# Establecemos la frontera en el punto medio
predicted <- ifelse(predicted > 0.5, 1, 0)
# Cálculo del error (métrica RMSE)
rmse(actual, predicted)