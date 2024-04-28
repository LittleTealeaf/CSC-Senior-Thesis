library(tidyverse)
library(ggrepel)


df <- read.csv("out/results/data.csv")

h <-9
w <- 12
h1 <- 4
w1 <- 5

plot <- ggplot(
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
    )

ggsave(
  plot = plot,
  file = "thesis/svg/variables.svg",
  height = h,
  width = w,
)

ggsave(
  plot = plot,
  file = "out/graphs/variables.png",
  height = h1,
  width = w1,
)

plot <- ggplot(
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
    )

ggsave(
  plot = plot,
  file = "thesis/svg/bootstraps.svg",
  height = h,
  width = w,
)



ggsave(
  plot = plot,
  file = "out/graphs/bootstraps.png",
  height = h1,
  width = w1,
)
