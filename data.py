import polars as pl 

df = pl.read_parquet("vocab.parquet")

# Filter out rows containing numbers and special symbols, keeping only letters and spaces
df = df.filter(
    pl.col("*").str.contains(r"^[a-zA-Z\s]*$")
)
print(len(df))
df.write_parquet("words.parquet")