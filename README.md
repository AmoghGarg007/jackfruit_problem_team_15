<body>
<h1>PES Jackfruit Problem</h1>
<p>
A Python-based graphical user interface for interactive image processing, enabling users to apply selected transformations with integrated undo, redo, and cropping functionalities, developed using Tkinter, OpenCV, PIL, and NumPy.
  <br><br>
</p>

<h2>Image Processing (Python)</h2>
<p>
This project is a Python-based image processing application developed as an academic mini project. It provides a graphical user interface built with Tkinter that allows users to apply image transformations interactively, with support for cropping and undo/redo operations. The project demonstrates the use of Python programming, OpenCV, PIL, and NumPy for basic image processing and GUI-based application development.
<br><br>
</p>
 
<h2>Features</h2>
<h3>⚫⚪ Grayscale</h3>
<ul>
  <li>Converts the loaded image from RGB color space to grayscale representation</li>
  <li>Reduces visual information to intensity values, highlighting structure and contrast</li>
  <li>Useful for preprocessing tasks and visual analysis where color information is not required</li>
</ul>
<p><br></p>

<h3>🔄 Flip Horizontal & Vertical</h3>
<ul>
  <li>Provides image reflection along the horizontal or vertical axis</li>
  <li>Allows users to correct orientation or create mirrored visual effects</li>
  <li>Implemented using pixel-wise transformations while preserving image quality</li>
</ul>
<p><br></p>

<h3>🔃 Invert Colors</h3>
<ul>
  <li>Inverts the RGB color values of the image to produce a negative effect</li>
  <li>Enhances visibility of details in certain lighting conditions</li>
  <li>Commonly used for artistic effects and contrast-based analysis</li>
</ul>
<p><br></p>

<h3>✏️ Sketch Effect</h3>
<ul>
  <li>Transforms the image into a pencil-sketch style representation</li>
  <li>Uses grayscale conversion, image inversion, Gaussian blurring, and color dodge blending</li>
  <li>Emphasizes edges and contours, simulating a hand-drawn sketch appearance</li>
</ul>
<p><br></p>

<h3>🎨 Cartoonify</h3>
<ul>
  <li>Applies a cartoon-style transformation to the image</li>
  <li>Combines color quantization, edge detection, and smoothing techniques</li>
  <li>Produces a simplified image with bold edges and reduced color palette, mimicking cartoon visuals</li>
</ul>
<p><br></p>

<h3>🖌️ Image Filters</h3>
<ul>
  <li>Provides multiple visual enhancement filters including Warm, Cool, Vintage, and Sharpen</li>
  <li>Adjusts color balance, saturation, and contrast to achieve different aesthetic effects</li>
  <li>Utilizes convolution, color space transformations, and histogram enhancement techniques</li>
</ul>
<p><br></p>

<h3>↩️ Undo Functionality</h3>
<ul>
  <li>Allows users to revert the image to a previous processing state</li>
  <li>Implemented using a stack-based approach to store past image states</li>
  <li>Ensures non-destructive editing during the image processing workflow</li>
</ul>
<p><br></p>

<h3>↪️ Redo Functionality</h3>
<ul>
  <li>Restores the most recently undone image operation</li>
  <li>Maintains a separate stack to track reverted states</li>
  <li>Enhances usability by allowing flexible experimentation with effects</li>
</ul>
<p><br></p>

<h3>✂️ Image Cropping</h3>
<ul>
  <li>Enables users to select and crop a specific region of interest from the image</li>
  <li>Provides real-time visual feedback through a preview window</li>
  <li>Supports precise image editing by mapping canvas selection to original image coordinates</li>
</ul>
<p><br></p>

  <h2>Modules Used</h2>
  <ol>
  <li>tKinter - Used to create the graphical user interface components such as windows, buttons, frames, and labels./li>
  <li>Pillow (PIL) - Handles image loading, saving, resizing, format conversion, and basic image manipulation.</li></li>
  <li>NumPy – Enables efficient numerical operations and array manipulation for image data processing.</li>
  <li>OpenCV (cv2) – Performs advanced image processing tasks such as filtering, edge detection, sketch, and cartoon effects.</li>
  </ol>
  <p><br></p>
  
  <h2>Project Highlights</h2>
<ul>
  <li>Well-structured design with clear separation of features</li>
  <li>Demonstrates practical applications of image processing techniques</li>
  <li>Allows users to interactively modify images using multiple transformations</li>
  <li>Supports saving and restoring image states during editing</li>
  <li>Simple and intuitive interface designed for ease of use</li>
</ul>


  <h2>How to Run the Project</h2>
  <ol>
  <li>
  Install all the required libraries (Mentioned in Requirements.txt) onto the system using

    pip install <module_name>
  </li>
  
  <li>Run the program:</li>
  
    python JackFruitProblemTeam15.py
  </ol>
  <h2>Team Project</h2>
  <p>This project was developed collaboratively as part of an academic mini project to demonstrate Python programming concepts and Image Processing functions using different Modules.<br><br>
  Team members:     <br>
  <ul>
  <li>Amogh - PES2UG25CS059</li>
  <li>Albin - PES2UG25CS053</li>
  <li>Akul  - PES2UG25CS051</li>
  <li>Anand - PES2UG25CS064</li>
  </ul>
  </p>
  <br>
  </body>

<h2>Photos</h2>
<p>Main Screen</p>
<img src="https://i.ibb.co/KjwqPyd6/Main-Screen.png" alt="Main-Screen" border="0">
<br><br>

<p>Main Screen (But Dark :D)</p>
<img src="https://i.ibb.co/6RzSTB8d/Main-Screen-Dark-Mode.png" alt="Main-Screen-Dark-Mode" border="0">
<br><br>

<p>Image Processing - Grayscale</p>
<img src="https://i.ibb.co/27Gq7QTj/Img-Process-Grayscale.png" alt="Img-Process-Grayscale" border="0">
<br><br>

<p>Image Processing - Cartoonify</p>
<img src="https://i.ibb.co/yFCh51sn/Img-Process-Cartoonify.png" alt="Img-Process-Cartoonify" border="0">
<br><br>

<p>Filter - Cool</p>
<img src="https://i.ibb.co/BHcxVzhR/Filter-Cool.png" alt="Filter-Cool" border="0">
<br><br>

<p>Tool - Cropping</p>
<img src="https://i.ibb.co/9Ht6hvBz/Cropping-Tool.png" alt="Cropping-Tool" border="0">
<br><br>

<h2>Sample Input/Output</h2>

<p>Sample input photo</p>
<img src="https://i.ibb.co/NdNrgpH3/Sample-Input.jpg" alt="Sample-Input" border="0">
<br><br>

<p>Opening the image onto the GUI to process</p>
<img src="https://i.ibb.co/JjwxgJ1t/Opening-The-Image.png" alt="Opening-The-Image" border="0">
<br><br>

<p>Saving the processed image (After applying Sketch Effect)</p>
<img src="https://i.ibb.co/dJjK4DF7/Saving-The-Processed-Image.png" alt="Saving-The-Processed-Image" border="0">
<br><br>

<p>The final processed image</p>
<img src="https://i.ibb.co/xqQ0mfR2/Processed-Image.jpg" alt="Processed-Image" border="0">

