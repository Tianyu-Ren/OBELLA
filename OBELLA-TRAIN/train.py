import torch
from model import AnswerEvaluationModel, DotProductAttention, masked_pooling, model_prediction, model_evaluate
from torch import nn
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
from load_data import load_training_dataset, load_test_dataset, load_validation_dataset
from utils import try_all_gpus, data_loading


def train(model, tokenizer, train_iter, valid_iter, max_length, epoch, num_steps, warmup_ratio, lr,
          accumulation_steps, optimizer=torch.optim.Adam, criterion=nn.CrossEntropyLoss):
    tokenizer_ = AutoTokenizer.from_pretrained(args.encoder_state)
    loss_table, valid_table = [], []
    global_step = 0
    update_step = 0
    best_dev_accuracy = 0

    # Set the device
    devices = try_all_gpus()
    device = devices[0]
    num_gpus = len(devices)

    # Set the optimizer
    optimizer = optimizer(model.parameters(), lr=lr)

    # Set the Loss
    criterion = criterion()

    # Set the schedule
    if num_steps == -1:
        num_steps = len(train_iter) * epoch // accumulation_steps
    print("Total Update Step: {}".format(num_steps))
    if warmup_ratio != -1.0:
        warmup_steps = int(warmup_ratio * num_steps)
        schedule = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                   num_training_steps=num_steps)
    else:
        schedule = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1,
                                                   num_training_steps=num_steps)

    # Check the model device
    if next(model.parameters()).device != device and num_gpus == 1:
        model = model.to(device)
    else:
        model = nn.DataParallel(model, device_ids=devices).cuda()

    # Start training
    scalar = torch.cuda.amp.GradScaler()
    model.train()
    for _ in range(epoch):
        for x in tqdm(train_iter):
            question, reference, candidate, label = x["question"], x["reference"], x["candidate"], x["label"]
            y_true = torch.tensor(label, dtype=torch.long, device=device)
            with torch.cuda.amp.autocast():
                y_hat = model_prediction(question, reference, candidate, model, tokenizer, max_length, device)
                loss = criterion(y_hat, y_true)
            scalar.scale(loss).backward()
            global_step += 1
            if global_step >= accumulation_steps and global_step % accumulation_steps == 0:
                update_step += 1
                schedule.step()
                scalar.step(optimizer)
                scalar.update()
                optimizer.zero_grad()
                loss_table.append(loss.item())
                if update_step % 5 == 0:
                    print(f"Current step: {update_step}, Loss: {loss.item()}")
            # Evaluate
            if global_step >= accumulation_steps * num_steps // 4 and global_step % 400 == 0:
                torch.cuda.empty_cache()
                _, accuracy = model_evaluate(model, tokenizer, valid_iter, max_length, device)
                model.train()
                valid_table.append(accuracy)
                print(f"Current step: {global_step}, Validation: {accuracy}")
                if accuracy > best_dev_accuracy:
                    torch.save(model.state_dict(), f"best_model_state")
                    best_dev_accuracy = accuracy
    # Final Check
    _, accuracy = model_evaluate(model, tokenizer, valid_iter, max_length, device)
    model.train()
    valid_table.append(accuracy)
    print(f"Current step: {global_step}, Validation: {accuracy}")
    if accuracy > best_dev_accuracy:
        torch.save(model.state_dict(), "best_model_state")
    # Save loss logging and validation logging
    loss_table = list(map(lambda x: str(x), loss_table))
    valid_table = list(map(lambda x: str(x), valid_table))
    with open("loss_record.txt", "w") as file:
        file.writelines(loss_table)
    with open("validation_record.txt", "w") as file:
        file.writelines(valid_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_state", type=str, default="alberta")
    parser.add_argument("--model_checkpoint", type=str, default="none")
    parser.add_argument("--num_class", type=int, default=3, help="The number of answer equivalence label")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--optimizer", default=torch.optim.AdamW)
    parser.add_argument("--criterion", default=torch.nn.CrossEntropyLoss)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=2)

    args = parser.parse_args()
    print(args)

    # Initialize Model
    model_ = AnswerEvaluationModel(args.encoder_state, DotProductAttention, masked_pooling, args.num_class)
    tokenizer_ = AutoTokenizer.from_pretrained(args.encoder_state)
    if args.model_checkpoint != "none":
        state_dict = torch.load(args.model_checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_.load_state_dict(state_dict)
        print("Checkpoint Loaded")
    print("Model Loaded Successfully")

    # Initialize Dataset
    train_set = load_training_dataset(args.seed)
    valid_set = load_validation_dataset()
    test_set = load_test_dataset()
    train_iter = data_loading(train_set, args.batch_size)
    valid_iter = data_loading(valid_set, args.test_batch_size)
    test_iter = data_loading(test_set, args.test_batch_size)

    print("Dataset Loaded Successfully")

    # Set train arguments
    train(model_, tokenizer_, train_iter, valid_iter, args.max_length, args.epoch, args.num_steps,
          args.warmup_ratio, args.lr, args.accumulation_steps, args.optimizer, args.criterion)

    print("Training completed")

    # Test
    try:
        state_dict = torch.load("best_model_state")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_.load_state_dict(state_dict)
    except:
        print("load_failure")
        exit()
    devices_ = try_all_gpus()
    device_ = devices_[0]
    _, accuracy = model_evaluate(model_, tokenizer_, test_iter, args.max_length, device_)
    print(accuracy)
