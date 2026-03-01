import csv
import os

os.makedirs('results', exist_ok=True)

infile = 'results/product.csv'
outfile = 'results/top_products.csv'

try:
    with open(infile, newline='', encoding='utf-8') as r, open(outfile, 'w', newline='', encoding='utf-8') as w:
        reader = csv.reader(r)
        writer = csv.writer(w)
        header = next(reader)
        writer.writerow(header)
        for i, row in zip(range(100), reader):
            writer.writerow(row)
    print(f"Wrote top 100 products to {outfile}")
except FileNotFoundError:
    print(f"Input file not found: {infile}")
except Exception as e:
    print(f"Error: {e}")
