# Face-Studio
A streamlit app which uses Instance Segmentation for Coloring Lips and Hair

## Features
- Lip Coloring<br>
- Hair Coloring<br>
- Eyeliner<br>

## Output
### Lip Coloring

![lips](https://user-images.githubusercontent.com/52783742/145670516-7f269055-ca21-4815-af74-046551e7b69b.png)

### Hair Coloring

![hair](https://user-images.githubusercontent.com/52783742/145670538-ed6f38a7-c29c-43c1-8dfe-098bdcdcc78d.png)

### Eyeliner

![eye](https://user-images.githubusercontent.com/52783742/145670735-aa07435f-a641-4de9-8718-6fa84477a928.png)


## Installation
### Python 3
- Clone the repo using ```git clone <repo_url>``` (Note: GitBash should be installed in your system)
- Create a virtual environment(using pip or anaconda) with ```python=3.6.13```
- Activate your environment
- Install Cmake and Dlib seperately (More information given in notes)
- Install the requirements.txt using ```pip install -r requirements.txt```
- Now from the CLI(Command Line Interface), navigate to the cloned folder
- Enter ```streamlit run app.py``` in CLI to play with the webapp

### Docker
- Install Docker Desktop and configure it for your OS as per instruction on Docker Web Page.
- To build the image, <br>
  ```docker build -t <APP-NAME>:latest . ```
- To run the docker image, <br>
  ```docker run -p 8501:8501 <APP-NAME>:latest```
- If browser is not opening, Open Docker Desktop, You can see a container running with your ```<APP-NAME>```. Click on Open in Browser option present with the container.

## Deployment
<a href = "https://facestudio.herokuapp.com">Click Here</a> to view the deployment.

## Notes
### Python 3
- If you have trouble installing ```dlib```, <a href="https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/"> Click Here</a> for a comprehensive installation guide.
- If your OS is Windows, then install Visual Studio Installer and in it <br>```Installed > Visual Studio Community > Modify > Desktop Development with C++ > C++ Cmake tools for Windows```<br> and install it. Then proceed to the installation instructions.
- In case you get an error installing dlib after the above step, try to install the packages in ```requirements.txt``` one by one with giving cmake precedence to dlib.
### Docker
- ```<APP-NAME>``` is the name of the app you are going to give to the docker image being built and the same app name is used for running the container.
