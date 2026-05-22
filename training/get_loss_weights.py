
def get_loss_weights (epoch, total_epoch):

   progress = epoch / total_epoch

   # 1° INTERVALLO (0% -> 10%): L'Actor resta 1.0, il Refiner sale dolcemente da 0.5 a 1.0
   if progress <= 0.1:
      alpha = progress / 0.1  # Va da 0.0 a 1.0 in questo intervallo
      lambda_actor = 1.0
      lambda_refiner = 0.5 + (alpha * 0.5)

   # 2° INTERVALLO (10% -> 50%): L'Actor scende a 0.5, il Refiner sale a 2.0
   elif progress <= 0.5:
      alpha = (progress - 0.1) / 0.4  # Va da 0.0 a 1.0 in questo intervallo
      lambda_actor = 1.0 - (alpha * 0.5)
      lambda_refiner = 1.0 + (alpha * 1.0)

   # 3° INTERVALLO (50% -> 100%): I valori restano stabili per la seconda metà
   else:
      lambda_actor = 0.5
      lambda_refiner = 2.0

   return lambda_actor, lambda_refiner