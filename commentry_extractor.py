import pandas as pd

df = pd.read_csv("data/23_24_match_details.csv")
commentary_text = "\n".join(df['events'].astype(str).tolist())

commentary_file = open("/Users/sylvesteranthony/Documents/FC_LLM/data/datacommentary.txt","w")
commentary_file.write(commentary_text)
commentary_file.close()

