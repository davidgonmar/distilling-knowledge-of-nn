import argparse
from models import Teacher, Student
import torch
import torchvision
from pathlib import Path
import os
from dataclasses import dataclass
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(0)
DATAPATH = Path("data")

TEACHER_WEIGHTS = DATAPATH / "teacher.pth"
STUDENT_WEIGHTS = DATAPATH / "student.pth"
STUDENT_WEIGHTS_NO_DISTILLATION = DATAPATH / "student_no_distillation.pth"


def get_loaders(batch_size: int, train=True):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    ds = torchvision.datasets.MNIST(
        DATAPATH, download=True, train=train, transform=transform
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train)
    return dl


@dataclass
class Config:
    lrs: dict
    epochs: int
    batch_size: int


teacherconfig = Config(lrs={0: 0.01, 10: 0.001, 20: 0.0001}, epochs=50, batch_size=1024)
studentconfig = Config(
    lrs={0: 0.01, 10: 0.001, 20: 0.0001},
    epochs=50,
    batch_size=1024,
)


def get_dict(model: torch.nn.Module):
    # if is compiled, take _orig_mod
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model.state_dict()


@dataclass
class GeneralConfig:
    temperature: int
    hard_loss_weight: float


general_config = GeneralConfig(
    temperature=3,
    hard_loss_weight=0.2,  # in the paper, they mentioned lower loss for hard labels
)


def parse_args():
    parser = argparse.ArgumentParser(description="Training Loop for Models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["teacher", "student"],
        default="teacher",
        help="teacher or student",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "train_no_distil", "test_no_distil"],
        default="train",
        help="train or test",
    )
    return parser.parse_args()


def train_teacher():
    teacher = Teacher().to(device)
    teacher.train()
    optimizer = torch.optim.Adam(
        teacher.parameters(), lr=0
    )  # we will update the learning rate manually
    criterion = torch.nn.CrossEntropyLoss()
    train_dl = get_loaders(teacherconfig.batch_size)

    def step(model, x, y):
        yhat = model(x)
        loss = criterion(yhat, y)
        return loss

    step = torch.compile(step, fullgraph=True)
    for epoch in range(teacherconfig.epochs):
        start = time.time()
        if epoch in teacherconfig.lrs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = teacherconfig.lrs[epoch]

        for x, y in train_dl:
            # use mixed precision training
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            loss = step(teacher, x, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()} Time: {time.time() - start}")
    torch.save(get_dict(teacher), TEACHER_WEIGHTS)


def test_teacher():
    teacher = Teacher().to(device)
    teacher.eval()
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS))
    teacher = torch.compile(teacher, fullgraph=True)
    test_dl = get_loaders(teacherconfig.batch_size, train=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            yhat = teacher(x)
            _, predicted = torch.max(yhat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f"Accuracy: {correct / total}")


def train_student():
    teacher = Teacher().to(device)
    if not os.path.exists(TEACHER_WEIGHTS):
        raise ValueError("Teacher weights not found. Please train the teacher first!")
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS))

    student = Student().to(device)
    optimizer = torch.optim.Adam(
        student.parameters(), lr=0
    )  # we will update the learning rate manually

    # as described in the paper, we use the teacher to generate the soft labels
    def get_loss_criterion():
        csehard = torch.nn.CrossEntropyLoss()

        def cse_with_temperature(student_output, teacher_output):
            # Scale the logits by the temperature
            student_output = student_output / general_config.temperature
            teacher_output = teacher_output / general_config.temperature

            # we need to normalize the teacher logits to get the soft labels
            soft_labels = torch.nn.functional.softmax(teacher_output, dim=1)
            return torch.nn.functional.cross_entropy(student_output, soft_labels)

        def loss(teacher_output, student_output, y):
            hard_loss = csehard(student_output, y)
            soft_loss = cse_with_temperature(student_output, teacher_output)
            return (
                general_config.hard_loss_weight * hard_loss
                + (1 - general_config.hard_loss_weight) * soft_loss
            )

        return loss

    criterion = get_loss_criterion()

    train_dl = get_loaders(studentconfig.batch_size)
    # freeze teacher
    teacher.eval()

    def step(model, x, y):
        student_output = model(x)
        teacher_output = teacher(x)
        loss = criterion(teacher_output, student_output, y)
        return loss

    step = torch.compile(step, fullgraph=True)
    for epoch in range(studentconfig.epochs):
        start = time.time()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            if epoch in studentconfig.lrs:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = studentconfig.lrs[epoch]
            optimizer.zero_grad()
            loss = step(student, x, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()} Time: {time.time() - start}")

    torch.save(get_dict(student), STUDENT_WEIGHTS)


def test_student():
    student = Student()
    student.eval()
    student.load_state_dict(torch.load(STUDENT_WEIGHTS))
    student = torch.compile(student, fullgraph=True)
    test_dl = get_loaders(studentconfig.batch_size, train=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_dl:
            yhat = student(x)
            _, predicted = torch.max(yhat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f"Accuracy: {correct / total}")


def test_student_no_distillation():
    student = Student()
    student.eval()
    student.load_state_dict(torch.load(STUDENT_WEIGHTS_NO_DISTILLATION))
    student = torch.compile(student, fullgraph=True)
    test_dl = get_loaders(studentconfig.batch_size, train=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_dl:
            yhat = student(x)
            _, predicted = torch.max(yhat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f"Accuracy: {correct / total}")


# for comparison, train student without distillation
def train_student_no_distillation():
    student = Student().to(device)
    optimizer = torch.optim.Adam(
        student.parameters(), lr=0
    )  # we will update the learning rate manually
    criterion = torch.nn.CrossEntropyLoss()
    train_dl = get_loaders(studentconfig.batch_size)

    def step(model, x, y):
        yhat = model(x)
        loss = criterion(yhat, y)
        return loss

    step = torch.compile(step, fullgraph=True)
    for epoch in range(studentconfig.epochs):
        start = time.time()
        if epoch in studentconfig.lrs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = studentconfig.lrs[epoch]

        for x, y in train_dl:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            loss = step(student, x, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item()} Time: {time.time() - start}")
    torch.save(get_dict(student), STUDENT_WEIGHTS_NO_DISTILLATION)


def main(args):
    if args.model == "teacher" and args.mode == "train":
        train_teacher()
    if args.model == "teacher" and args.mode == "test":
        test_teacher()
    if args.model == "student" and args.mode == "train":
        train_student()
    if args.model == "student" and args.mode == "test":
        test_student()
    if args.model == "student" and args.mode == "train_no_distil":
        train_student_no_distillation()
    if args.model == "student" and args.mode == "test_no_distil":
        test_student_no_distillation()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train_no_distil":
        assert args.model == "student", "train_no_distil is only for student model"
    if args.mode == "test_no_distil":
        assert args.model == "student", "test_no_distil is only for student model"
    main(args)
