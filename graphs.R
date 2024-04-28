library(tidyverse)
library(ggrepel)


df <- read.csv("out/results/data.csv")

ggsave(
  plot = ggplot(
    data = df %>%
      mutate(Model = model, time = time / 1e6),
    mapping = aes(x = variables, y = time, color = Model)
  ) +
    geom_smooth(se = FALSE) +
    scale_y_continuous(trans = "log10") +
    xlab("Variables") +
    ylab("Time (milliseconds)") +
    theme_bw() +
    theme(
      legend.position = c(0.7, 0.25),
    ) +
    ggtitle("Average Time by Variable Dimension"),
  file = "out/graphs/variables.svg",
  height = 5,
  width = 5,
)


ggsave(
  plot = ggplot(
    data = df %>%
      mutate(Model = model, time = time / 1e6),
    mapping = aes(x = bootstraps, y = time, color = Model)
  ) +
    geom_smooth(se = FALSE) +
    scale_y_continuous(trans = "log10") +
    xlab("Observations") +
    ylab("Time (milliseconds)") +
    theme_bw() +
    theme(
      legend.position = c(0.7, 0.25),
    ) +
    ggtitle("Average Time by Observation Count"),
  file = "out/graphs/bootstraps.svg",
  height = 5,
  width = 5,
)
