library(tidyverse)

df_cuda <- read.csv("out/results/cuda/times.csv") %>% mutate(Model = "CUDA")
df_python_tf_gpu <- read.csv("out/results/python-tf-gpu/times.csv") %>%
  mutate(Model = "Tensorflow GPU")
df_python_tf_cpu <- read.csv("out/results/python-tf-cpu/times.csv") %>%
  mutate(Model = "Tensorflow CPU")
df_rust <- read.csv("out/results/rust/times.csv") %>%
  mutate(Model = "Rust")

df <- rbind(df_cuda, df_python_tf_cpu) %>%
  rbind(df_python_tf_gpu) %>%
  rbind(df_rust)


plot <- ggplot(
  data = df %>%
    group_by(variables, Model) %>%
    summarize(time = mean(time)),
  mapping = aes(x = variables, y = time, color = Model)
) +
  geom_point() +
  geom_smooth() +
  xlab("Variable Count") +
  ylab("Time")

ggsave(file = "out/graphs/variables.png", plot = plot, width = 10, height = 10)


plot <- ggplot(
  data = df %>%
    group_by(bootstraps, Model) %>%
    summarize(time = mean(time)),
  mapping = aes(x = bootstraps, y = time, color = Model)
) +
  geom_point() +
  geom_smooth() +
  xlab("Number of Observations") +
  ylab("Time")

ggsave(file = "out/graphs/bootstraps.png", plot = plot, width = 10, height = 10)
