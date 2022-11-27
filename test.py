from app import cli
from recsys import utils, data

cli.optimize()
cli.train_model_app()


item_id = 122 
top_k = 10
recomended = cli.recommendation(item_id=item_id, top_k=top_k)
print(recomended)

#print(cli.predict())

