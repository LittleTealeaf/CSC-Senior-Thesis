library(tidyverse)


df <- read.csv("out/results/data.csv")

plot <- ggplot(
  data = df %>%
  group_by(variables, model) %>%
  summarize(time = mean(time)),
  mapping = aes(x = variables, y = time, color = model)
) +
  geom_smooth() +
  geom_point()

ggsave(file = "out/graphs/variables.png", plot = plot, height = 10, width =10)


plot <- ggplot(
  data = df %>%
    group_by(bootstraps, model) %>%
    summarize(time = mean(time)),
  mapping = aes(x = bootstraps, y = time, color = model)
) +
  geom_point() +
  geom_smooth() +
  xlab("Number of Observations") +
  ylab("Time")

ggsave(file = "out/graphs/bootstraps.png", plot = plot, width = 10, height = 10)

