import polars as pl 

df = pl.read_parquet("vocab.parquet")
df = df.filter(
    pl.col("*").str.contains(r"^[a-zA-Z\s]*$")
)
print(len(df))
df.write_parquet("words.parquet")