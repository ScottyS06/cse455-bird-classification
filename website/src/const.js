const DATASET_SPLIT_CODE = `# Generate train, val, and test datasets
trainset = torchvision.datasets.ImageFolder(root='birds21wi/train', transform=transform_train)
testset = torchvision.datasets.ImageFolder(root='birds21wi/testing', transform=transform_test)

# 90:10 split of training set to generate a validation set
train_split, val_split = random_split(trainset, [int(0.9 * len(trainset) + 1), int( 0.1 * len(trainset))])`;

const DATA_TRANSFORMS = `# Define transformations for train set
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])`;

const options = {
  responsive: true,
  plugins: {
    legend: {
      position: "top",
    },
    title: {
      display: true,
      text: "Resnet50 (v2) CrossEntropy Loss vs. Epoch",
    },
  },
  scales: {
    y: {
      title: {
        display: true,
        text: "Loss",
      },
    },
    x: {
      title: {
        display: true,
        text: "Epoch",
      },
    },
  },
};

const resnet_options = {
  ...options,
  plugins: {
    title: {
      display: true,
      text: "Resnet50 (v2) CrossEntropy Loss vs. Epoch",
    },
  },
};

const efficientnet_options = {
  ...options,
  plugins: {
    title: {
      display: true,
      text: "EfficientNet_v2 CrossEntropy Loss vs. Epoch",
    },
  },
};

const labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];

const resnet_data = {
  labels,
  datasets: [
    {
      label: "Training",
      data: [
        2.988625919994186, 1.6375558893470203, 0.716428503722829,
        0.4927413358407862, 0.4109168100882979, 0.39446640420047674,
        0.330974789117189, 0.29359552850398946, 0.2519773317172247,
        0.2254188564222525,
      ],
      borderColor: "rgb(255, 99, 132)",
      backgroundColor: "rgba(255, 99, 132, 0.5)",
    },
    {
      label: "Validation",
      data: [
        5.2447486295080035, 2.9068508296282802, 0.9140272367201736,
        0.8268557038303399, 0.8057318879123264, 0.39456819982093627,
        0.3859594821677635, 0.2965309173431549, 0.31585906516155987,
        0.3094830174473708,
      ],
      borderColor: "rgb(53, 162, 235)",
      backgroundColor: "rgba(53, 162, 235, 0.5)",
    },
  ],
};

const efficientnet_data = {
  labels,
  datasets: [
    {
      label: "Training",
      data: [
        3.7846116009777204, 2.2912635168757007, 1.179815260635116,
        0.86723507230014, 0.7354046673296126, 0.6506264851347957,
        0.5796684876331308, 0.5250873504751954, 0.4784406052549021,
        0.452018068969579,
      ],
      borderColor: "rgb(255, 99, 132)",
      backgroundColor: "rgba(255, 99, 132, 0.5)",
    },
    {
      label: "Validation",
      data: [
        2.905156659742708, 2.4199750984443504, 1.1521775672467718,
        1.0653295708039627, 0.9873102972689115, 0.940639273161211,
        0.8942570020727163, 0.9214688264638045, 0.9132831874661128,
        0.8960600497791208,
      ],
      borderColor: "rgb(53, 162, 235)",
      backgroundColor: "rgba(53, 162, 235, 0.5)",
    },
  ],
};

export const Constant = {
  DATASET_SPLIT_CODE,
  DATA_TRANSFORMS,
  efficientnet_data,
  resnet_options,
  efficientnet_options,
  resnet_data,
};
