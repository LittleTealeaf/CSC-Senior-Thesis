library(tidyverse)

df_cuda <- read.csv("out/cuda/times.csv") %>% mutate(Model = "CUDA")
df_python_tf_gpu <- read.csv("out/python-tf-gpu/times.csv") %>%
  mutate(Model = "Tensorflow GPU")
df_python_tf_cpu <- read.csv("out/python-tf-cpu/times.csv") %>%
  mutate(Model = "Tensorflow CPU")
df_rust <- read.csv("out/rust/times.csv") %>%
  mutate(Model = "Rust")

df <- rbind(df_cuda, df_python_tf_gpu) %>%
  rbind(df_python_tf_cpu) %>%
  rbind(df_rust)


plot <- ggplot(
  data = df %>% filter(epoch > 0),
  mapping = aes(x = epoch, y = time, color = Model)
) +
  geom_point() +
  scale_y_continuous(
    labels = scales::label_number(suffix = "ns", big.mark = ",")
  ) +
  xlab("Iteration") +
  ylab("Time")
ggsave(file = "out/img/graph.png", plot = plot, width = 10, height = 10)
