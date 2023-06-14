import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from util.util_logger import val_log_saver
from util.util_model_saver import save_model


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
   
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (img, tab, y) in enumerate(dataloader):
        # Send data to target device
        img, tab, y = img.to(device), tab.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(img, tab)

        # Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        for batch, (img, tab, y) in enumerate(dataloader):
            
            img, tab, y = img.to(device), tab.to(device), y.to(device)

            y_pred = model(img, tab)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            test_pred_labels = y_pred.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          log_txt_saver: bool,
          save_model: bool,
          test_name: str) -> Dict[str, List]:
    
  # Create empty results dictionary
  results = {"train_loss": [],
              "train_acc": [],
              "test_loss": [],
              "test_acc": []
  }
  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                      dataloader=train_dataloader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      device=device)
    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

    # Print out what's happening
    print(
      f"Epoch: {epoch+1} | "
      f"train_loss: {train_loss:.4f} | "
      f"train_acc: {train_acc:.4f} | "
      f"test_loss: {test_loss:.4f} | "
      f"test_acc: {test_acc:.4f}"
    )

    # Update results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  if log_txt_saver == True:
    val_log_saver(test_name, results,"train_loss")
    val_log_saver(test_name, results,"train_acc")
    val_log_saver(test_name, results,"test_loss")
    val_log_saver(test_name, results,"test_acc")

  if save_model:
    save_model(model=model,
              target_dir="models",
              model_name=f"{test_name}.pth")

  # Return the filled results at the end of the epochs
  return results