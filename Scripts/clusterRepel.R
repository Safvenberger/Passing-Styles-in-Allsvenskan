library(ggplot2)
library(ggrepel)
library(stringr)
library(extrafont)
library(showtext)

# Specify the working directory
if(length(str_split(getwd(), "/Scripts")[[1]]) != 2){
  setwd("Scripts")
}

# Import extra font
font_add("Roboto", "../Fonts/Roboto-Black.ttf")
showtext_auto()

# Read the data
plot_data <- read.csv("../Data/plot_data.csv")

# Specify a colorblind friendly palette
cbPalette <- c('#4477AA', '#882255', '#228833', '#EE7733', '#EE3377', '#DDAA33', 
               '#BBBBBB', '#88CCEE')

# Create the plot
ggplot(plot_data, aes(PC1, PC2, 
                      color = factor(cluster, 
                                     labels=c("1: The final third creator", "2: The goalkeeper", 
                                              "3: The defensive outlet", "4: The target man", 
                                              "5: The unwilling passer", "6: The winger", 
                                              "7: The defensive passer", "8: The advanced playmaker")), 
                     label = player)) +
  geom_point() + geom_text_repel(show.legend = FALSE, max.overlaps = 8, size=10) + 
  scale_color_manual(name = "", values = cbPalette) +
  theme_classic() + theme(legend.position = "top", panel.grid = element_blank(),
                          axis.line.x.bottom = element_line(color="#C0C0C0"),
                          axis.line.y.left =  element_line(color="#C0C0C0"),
                          axis.text = element_text(size=30, color="black"), 
                          axis.title = element_text(size=30),
                          legend.text = element_text(size=30)) + 
  guides(label = "none")

# Save the figure
ggsave("../Figures/cluster_names_repel.png", width = 10, height = 6)

