import torch
import matplotlib.pyplot as plt

def train_model(net, train_loader, optimizer, scheduler, loss_function, num_epochs, device, class_names, batch_size):
    loss_history = []

    for e in range(num_epochs):
        print(f"Epoch: {e+1}/{num_epochs}")

        epoch_loss = 0
        correct = 0
        total_examples = 0

        # Check to ensure train_loader is working properly
        if len(train_loader) == 0:
            print("Error: train_loader is empty!")
            break

        for batch_idx, (X, y) in enumerate(train_loader):
            # Ensure that batches are being retrieved and have the correct size
            if X.size(0) != batch_size and batch_idx != len(train_loader) - 1:
                print(f"Warning: Batch size mismatch at batch {batch_idx}, expected {batch_size} but got {X.size(0)}")
            
            # Move data to the device
            X = X.to(device)
            y = y.to(device)

            # Forward pass
            outputs = net(X)
            batch_loss = loss_function(outputs, y)  # Calculate the loss

            # Backward pass
            batch_loss.backward()

            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()

            # Track the epoch loss
            epoch_loss += batch_loss.item()

            # Print status every 50 batches
            if batch_idx % 50 == 0:
                tot_images = batch_idx * batch_size + len(X)
                print(f"\tIteration: {batch_idx}\t Current batch Loss: {batch_loss:.3f} | images: {tot_images} | epoch loss/n: {epoch_loss/len(train_loader):.3f}")

            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            compare = y == preds
            correct += torch.sum(compare)
            total_examples += len(compare)

        # Step the learning rate scheduler
        scheduler.step()

        # Print epoch summary
        accuracy = (correct / total_examples).item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"  Train epoch loss: {epoch_loss:.2f} | mean: {avg_epoch_loss:.2f} | accuracy: {accuracy:.2f} | lr: {scheduler.get_last_lr()}")
        loss_history.append(avg_epoch_loss)

    print("Done!")
    return loss_history
