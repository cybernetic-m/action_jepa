
def get_loss_weights (epoch, total_epoch):
  
  progress = epoch/total_epoch

  if progress < 0.3:
     lambda_actor, lambda_refiner = 1.0, 0.1
  elif progress < 0.5:
     alpha = (progress - 0.3) / 0.2 # coefficent that goes from 0 to 1 in this interval
     lambda_actor = 1.0 - (alpha*0.9)
     lambda_refiner = 0.1 + (alpha*0.9)
  else:
     lambda_actor, lambda_refiner = 0.1, 1.0
  return lambda_actor, lambda_refiner