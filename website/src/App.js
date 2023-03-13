import "./App.css";
import React, { useState } from "react";
import styled from "styled-components";
import { Constant } from "./const";

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

function App() {
  return (
    <>
      <Header>
        <h1>CSE455 Final Project</h1>
        <h2>Bird Classification</h2>
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
        <Link href="https://www.kaggle.com/competitions/birds23wi/data">
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
        <SectionSubHeader>Approach / Techniques Used</SectionSubHeader>
        <Description>
          Since the test dataset that was provided did not include labels for
          the images it was difficult to see how well the model was generalizing
          and performing on unseen data. One approach would have been to submit
          the models predictions to the kaggle competition everytime I wanted to
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
          read files every time the __get_item__ function was called. This also,
          slowed down the training process and thus, I decided to abandon this
          method and switch over to Google Colab.
        </Description>
        <Description>
          I also performed some data augmentation to make the model generalize
          better. To do this, I used PyTorch's transformation module to perform
          perspective transformations and image flipping. However, I did not see
          great impact from the perspective transform.
        </Description>
        <CodeMirror value={Constant.DATA_TRANSFORMS} extensions={[python()]} />
      </Container>
      <Container>
        <SectionHeader>Training Techniques</SectionHeader>
        <Link>Github</Link>
        <SectionSubHeader>Transfer Learning</SectionSubHeader>
        For this classification task, I decided to use tranfer learning as the
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
          I decided to test these models because I wanted to have a good range
          of experiments. Also, from my preliminary resarch I found that Resnet
          models are good for transfer learning and offer a less complex
          architecture that is relatively fast compared to other models with
          over 100 million parameters. This was ideal because with the limited
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
          I found from my experiments, learning rate to be the strongest
          indicator on training performance and continued decrease loss. The
          approach I used was to set the learning rate to a larger value such as
          0.1 for the first few epochs and then decrease exponentially once loss
          began to plateau. Also, batch size played an important role in
          maintaining the model computation within the Google Colab provided
          resource limit. I had to tune this parameter to smaller values such as
          32 or 64 for larger models like EfficientNet and ConvNeXt.
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
            <td></td>
          </tr>
        </table>
        <SectionSubHeader>Resnet50 - v2</SectionSubHeader>
        <Line options={Constant.options} data={Constant.data} />
        <SectionSubHeader>EfficientNet</SectionSubHeader>
        <Line options={Constant.options} data={Constant.data} />
      </Container>
      <Container>
        <SectionHeader>Project Video</SectionHeader>
        <ProjectVideo
          title="Project Video"
          width="420"
          height="315"
          src="https://www.youtube.com/embed/tgbNymZ7vqY"
        ></ProjectVideo>
      </Container>
      <Container>
        <SectionHeader>Moving Forward</SectionHeader>
      </Container>
      <Container>
        <SectionHeader>How My Approach Differs From Others</SectionHeader>
      </Container>
    </>
  );
}

export default App;
