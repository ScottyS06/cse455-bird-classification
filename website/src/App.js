import "./App.css";
import React from "react";
import styled from "styled-components";
import { Constant } from "./const";
import transforms from './transforms.png';

import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const Header = styled.div`
  background-color: #41b096;
  padding: 10vh 0;
  text-align: center;
`;

const SectionHeader = styled.div`
  font-size: 1.5em;
  font-weight: bold;
  margin: 10px 0;
`;

const SectionSubHeader = styled.div`
  font-size: 1.2em;
  color: #666665;
  font-weight: bold;
  margin: 10px 0;
`;

const ListItem = styled.li`
  margin: 5px 0;
`;

const Container = styled.div`
  display: block;
  align-items: start;
  width: 70%;
  padding: 10px;
  margin: 30px auto 30px;
  margin-left: auto;
  margin-right: auto;
  border-radius: 0.75rem;
  box-shadow: 5px 5px 10px 2px rgba(0, 0, 0, 0.1);
`;

const Description = styled.p`
  display: block;
  line-height: 1.5rem;
`;

const Link = styled.a`
  text-decoration: underline;
  color: blue;
`;

const ProjectVideo = styled.iframe`
  display: block;
  width: 100%;
  height: 50vh;
`;

const TransformsImage = styled.img`
  width: 100%;
`

function App() {
  return (
    <>
      <Header>
        <h1>CSE455 Final Project</h1>
        <h2 style={{ fontWeight: 400 }}>Investigating ImageNet Models on Bird Classification</h2>
        <h4 style={{ fontWeight: "normal" }}>Group Members: Scotty Singh</h4>
      </Header>
      <Container>
        <SectionHeader>Project Description</SectionHeader>
        <Description>
          For my project, I decided to participate in the biannual bird
          classification challenge. The goal of this challenge was to create a
          classification model that would compete for the best validation
          accuracy on the provided test dataset containing 10,000 images. The
          challenge provided a large dataset of 555 species of birds with over
          40,000 images. For this task, I applied several techniques learned
          from class as well as those foundational to computer vision including
          data processing, convolutional neural networks, transfer learning, and
          hyperparameter tuning. I experimented with several pre-trained
          ImageNet models (Resnet18, Resnet50, EfficientNet) and faced several
          challenges along the way.
        </Description>
      </Container>
      <Container>
        <SectionHeader>Data Processing</SectionHeader>
        <Link href="https://www.Kaggle.com/competitions/birds23wi/data">
          Bird Classification Dataset
        </Link>
        <SectionSubHeader>Challenges</SectionSubHeader>
        <Description>
          The Birds dataset presented several challenges while developing an
          effective and accurate model. Some of these challenges include:
        </Description>
        <ol>
          <ListItem>
            The test dataset provided did not include labels for the images.
          </ListItem>
          <ListItem>
            Images from dataset are large and utilize lots of cpu power when
            loading and transforming.
          </ListItem>
        </ol>
        <SectionSubHeader id="DataProcessing">Approach / Techniques Used</SectionSubHeader>
        <Description>
          Since the test dataset that was provided did not include labels for
          the images it was difficult to see how well the model was generalizing
          and performing on unseen data. One approach would have been to submit
          the models predictions to the Kaggle competition every time I wanted to
          check the models performance but this was not a feasible or practical
          solution. Thus, I decided to split the training dataset into two sub
          datasets one for training and one for validation. I used a 90-10 split
          for training and validation.
        </Description>
        <CodeMirror
          value={Constant.DATASET_SPLIT_CODE}
          extensions={[python()]}
        />
        <Description>
          When starting this project, I initially decided to use Kaggle
          notebooks for training the model however I quickly ran into some
          difficulties where training was taking a really long time. After doing
          some research, I found out that the training was bottlenecked by the
          dataloaders having to resize and transform the images. Thus, I decided
          to try preprocessing the images by resizing and transforming them
          beforehand and saving the tensors instead of the actual images to
          speed up the dataloaders. However, the tensors themselves, being very
          large, started taking too much memory in the Kaggle notebook. Also,
          the custom dataset I wrote to load this data made OS calls to open and
          read files every time the __get_item__ function was called. This also
          slowed down the training process and thus, I decided to just resize the images and save them as jpegs.
        </Description>
        <Description>
          I also performed some data augmentation to make the model generalize
          better. To do this, I used PyTorch's transformation module to perform
          perspective transformations, image flipping, normalization, and rotations.
        </Description>
        <CodeMirror value={Constant.DATA_TRANSFORMS} extensions={[python()]} />
        <TransformsImage alt="transforms" src={transforms}></TransformsImage>
      </Container>
      <Container>
        <SectionHeader>Training Techniques</SectionHeader>
        <Link href="https://github.com/ScottyS06/cse455-bird-classification">Github</Link>
        <SectionSubHeader>Transfer Learning</SectionSubHeader>
        For this classification task, I decided to use transfer learning as the
        main approach to developing a high accuracy model. From class, I learned
        that pretrained models can be good feature extractors and can speed up
        the training process because they are already trained on a similar task.
        I decided to test the following models:
        <ol>
          <ListItem>Resnet18</ListItem>
          <ListItem>Resnet50 - v1 ImageNet weights</ListItem>
          <ListItem>Resnet50 - v2 ImageNet weights</ListItem>
          <ListItem>EfficientNet v2 </ListItem>
        </ol>
        <Description>
          I wanted to test a wide variety of models with various architectures.
          Also, from my preliminary research I found that Resnet
          models are good for transfer learning and offer a less complex
          architecture that is relatively fast compared to other models which
          can have over 100 million parameters. This was ideal because with the limited
          resources I had I wanted to use a model that would still perform
          relatively well. I decided to also test the EfficientNet model because
          it had a high top-1 accuracy and was a relatively small model with
          roughly 20 million parameters. I had also not seen many people use
          this model for transfer learning which was another reason I wanted to
          test it out. I also tried a few other models ConvNeXt and
          EfficientNet_b0 but decided to not continue training after poor
          initial performance.
        </Description>
        <SectionSubHeader>Learning Rate and Batch Size</SectionSubHeader>
        <Description>
          From my experiments I found learning rate to be the strongest
          indicator on training performance and continued decrease in loss. The
          approach I used was to set the learning rate to a larger value such as
          0.1 for the first few epochs and then decrease exponentially once loss
          began to plateau. Also, batch size played an important role in
          maintaining the model computation within the Google Colab and Kaggle provided
          resource limit. I had to tune this parameter to smaller values such as
          32 or 64 for larger models like EfficientNet and ConvNeXt.
        </Description>
        <SectionSubHeader id="OtherApproaches">Other Approaches (unsuccessful)</SectionSubHeader>
        <Description>
          I also tried using some more unique approaches to training in hopes of getting higher overall accuracy. These include the following:
          <ol>
            <ListItem>After training on smaller images (224x224) change dataset to large images (512x512)</ListItem>
            My reasoning behind this approach was that after training on smaller images which would likely help identify more high level features such as shape and color of the birds switching to a larger image may help pick out more fine grain details. However, this approach was very wrong. I used the Resnet50 model for this experiment and found that the loss greatly increased after switching to the larger images. This was likely because the model which was good at selecting features for 224x224 images may not be able to identify those features as well in a larger images because the features themselves may look different. Also, with larger images this technique was not effective due to the limited resources offered in Google Colab.
            <ListItem>Use only a training set with more data augmentation</ListItem>
            With this approach, I hoped to get the model to see more images of birds and thus perform better as well. However, again this approach was unsuccessful. The model began overfitting to the data and it was not possible to tell how the model was generalizing since there was no validation set.
          </ol>
        </Description>
      </Container>
      <Container>
        <SectionHeader>Experiment Results</SectionHeader>
        <SectionSubHeader>Model Test Set Performance</SectionSubHeader>
        <table>
          <tr>
            <th>Model</th>
            <th>Test Accuracy</th>
          </tr>
          <tr>
            <td>Resnet18</td>
            <td>0.507</td>
          </tr>
          <tr>
            <td>Resnet50 - v1</td>
            <td>0.701</td>
          </tr>
          <tr>
            <td>Resnet50 - v2</td>
            <td>0.7995</td>
          </tr>
          <tr>
            <td>EfficientNet - v2</td>
            <td>0.7795</td>
          </tr>
        </table>
        <SectionSubHeader>Resnet50 - v2</SectionSubHeader>
        <Line options={Constant.resnet_options} data={Constant.resnet_data} />
        <SectionSubHeader>EfficientNet</SectionSubHeader>
        <Line options={Constant.efficientnet_options} data={Constant.efficientnet_data} />
      </Container>
      <Container>
        <SectionHeader>Project Video</SectionHeader>
        <ProjectVideo
          title="Project Video"
          width="420"
          height="315"
          allow="fullscreen;"
          src="https://www.youtube.com/embed/J8AB16wwk8M"
        ></ProjectVideo>
      </Container>
      <Container>
        <SectionHeader>What problems did I encounter?</SectionHeader>
        <Description>
          <ol>
            <ListItem>Google Colab and Kaggle both had a limited amount of computational resources.</ListItem>
            This was one of the main challenges I faced. With the limited resources it was difficult to train very large models and I had to be especially mindful of the size of the images I was using for training as well as the batch size. The training process was also very slow so it was difficult to test out hypotheses quickly to determine if they were worth investigating further. This caused me to spend a lot of time on techniques that were not effective and did not improve the overall model performance. 
            <ListItem>Dataloaders do not work well with GPUs and can consume lots of CPU resources.</ListItem>
            Loading the data directly from dataloaders and performing transformations during the training process was computationally expensive and slow. As described in the <a href="#DataProcessing">Data Processing Section</a>, I tried resizing images and saving them as tensors, however, the custom dataset module I wrote did not load the data effectively and was still slow. I decided to then just resize the images offline keeping them in the same format as the provided Kaggle dataset.
            <ListItem>Test dataset did not contain labels.</ListItem>
            Without the test labels there was no way to tell how my models were performing on unseen data. Thus, I had to do some research into common practices to overcome this challenge. I read about K-Fold cross validation but this required running the model through several training loops and since training for a few epochs was already taking a along time, I deemed this method unfeasible. I decided to perform a simple random split of the data to generate a validation set.
          </ol>
        </Description>
      </Container>
      <Container>
        <SectionHeader>Moving Forward</SectionHeader>
        <Description>
          In the next steps of this project, I would like learn more about data processing. The constraints on how long I could train, the kind of model I could use, and the types of data augmentation that I could perform were largely affected by the dataset and my understanding of data processing. My initial approach to preprocessing the data and storing resized and transformed data seemed promising but due to time constraints and challenges I had to use a different method. I would like to explore more into this technique and possibly research into how other people reduce the time it takes to load images and how to make the process more efficient.
        </Description>
        <Description>
          I would also like to explore deeper into hyperparameters and model architectures. Though my model accuracy greatly increased from where I first started (~50% to ~80%), there were people who achieved close to 90% accuracy. I believe training longer could not have solely been the reason for this difference and would like to investigate further into possible causes. Maybe more data augmentation or normalization techniques would have helped my model generalize better. Possibly even collecting more data.
        </Description>
        <Description>
          Finally, I would also like to create a simple 4-5 layer convolutional neural network from scratch and see how well it would perform in comparison to the other pretrained models I used. I believe that training solely on the desired dataset could be beneficial as the model would be very good at one thing and may not have biases from other data sources. This is an interesting idea and could be worth investigating more.
        </Description>
      </Container>
      <Container>
        <SectionHeader>How My Approach Differs From Others</SectionHeader>
        <Description>
          My approach differs from others because I experimented with several model architectures and utilized unique data processing and augmentation techniques. Though most of these were unsuccessful I had not seen them be used in other Kaggle/Google Colab notebooks. Firstly, I tested 5 different pretrained models whereas most only trained 1 or 2. I experimented with multiple Resnet architectures, ConvNeXt, and EfficientNet. What I found was that more complex models performed better on the training dataset but were also more prone to overfitting. These models also took longer to train since they had far more parameters than simple models like Resnet18. I found that batch size was an important metric to tune when dealing with large models. I initially started with a batch size of 128 but had to decrease this to 32 to optimize my GPU usage in the constrained environment.
        </Description>
        <Description>
          Additionally, I utilized several data augmentation techniques. Specifically, I applied a Normalize transform to the images using the mean and standard deviation from the ImageNet dataset. I found that this technique greatly helped speed up training especially for the EfficientNet_v2 model in the early epochs during training. I was able to decrease loss from ~6.2 to ~1.2 within one epoch. Reading a few articles online, I found that this normalization technique works well since the birds dataset shares a similar image structure to many of the ImageNet samples.
        </Description>
        <Description>
          Another technique I used that I hadn't seen others do is to freeze half the layers of the pretrained model. I noticed this helped speed up training in the early epochs but hindered final model performance slightly. This is likely because the early layers are good feature extractors which can be applied to a variety of datasets whereas the later layers may be more specific to the ImageNet dataset. I also tried freezing all the layers and only training a final classification layer but found that this did not work well. Retraining all the weights of the model had the best output, specifically for my highest accuracy training setup (Resnet50 with v2 weights), but also took a very long time to train. This is because calculating the gradients for all the parameters is more computationally expensive than a subset of the parameters. This likely also performed well because the ImageNet weights are a good starting point to train the entire network. By training the entire network, the model became less generalized (likely would not perform well on ImageNet) but more suited for bird classification.
        </Description>
        <Description>
          Finally, I also tried a few other unusual approaches such as training on small images and then training further on larger images. I have included more information on these methods in the <a href="#OtherApproaches">Other Approaches Section</a>.
        </Description>
      </Container>
      <Container>
        <SectionHeader>Resources</SectionHeader>
        <ol>
          <li><a href="https://pytorch.org/vision/main/models.html">https://pytorch.org/vision/main/models.html</a><br />[Pretrained models from Pytorch Hub]</li>
          <li><a href="https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing#scrollTo=C5_LglWCs9Iu">https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing#scrollTo=C5_LglWCs9Iu</a><br />[Pytorch Tutorial from class for helper methods]</li>
          <li><a href="https://medium.com/codex/saving-and-loading-transformed-image-tensors-in-pytorch-f37b4daa9658">https://medium.com/codex/saving-and-loading-transformed-image-tensors-in-pytorch-f37b4daa9658</a><br />[Data processing article]</li>
        </ol>
      </Container>
    </>
  );
}

export default App;
