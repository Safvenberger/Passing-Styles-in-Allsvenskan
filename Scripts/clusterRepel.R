library(ggplot2)
library(ggrepel)
library(stringr)

# Specify the working directory
if(length(str_split(getwd(), "/Scripts")[[1]]) != 2){
  setwd("Scripts")
}

# Read the data
plot_data <- read.csv("../Data/plot_data.csv")

# Specify a colorblind friendly palette
cbPalette <- c('#4477AA', '#882255', '#228833', '#EE7733', '#EE3377', '#DDAA33', 
               '#BBBBBB', '#88CCEE')

# Create the plot
ggplot(plot_data, aes(PC1, PC2, color = factor(cluster), 
                     label = player)) +
  geom_point() + geom_text_repel(show.legend = FALSE) + 
  scale_color_manual(name = "Cluster", values = cbPalette) +
  theme_classic() + theme(legend.position = "top") + guides(label = "none")

ggsave("../Figures/cluster_names_repel.png", width = 12, height = 8)


plot_data %>% 
  group_by(cluster) %>% 
  slice_max(order_by = Market.value)
