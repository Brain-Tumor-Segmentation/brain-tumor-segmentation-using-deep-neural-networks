from ._anvil_designer import Form1Template
from anvil import *
import anvil.server
import plotly.graph_objects as go
import anvil.server
import anvil.image

class Form1(Form1Template):

  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run when the form opens.

  def flair_change(self, file, **event_args):
    """This method is called when a new file is loaded into this FileLoader"""
    ext = file.name.split("_")[-1]
    if ext != "flair.nii.gz":
      self.error_msg.visible = True
      self.error_msg.text = "Upload the correct FLAIR file"
    else:
      self.error_msg.visible = False

  def t1_change(self, file, **event_args):
    """This method is called when a new file is loaded into this FileLoader"""
    ext = file.name.split("_")[-1]
    if ext != "t1.nii.gz":
      self.error_msg.visible = True
      self.error_msg.text = "Upload the correct T1 file"
    else:
      self.error_msg.visible = False

  def t1ce_change(self, file, **event_args):
    """This method is called when a new file is loaded into this FileLoader"""
    ext = file.name.split("_")[-1]
    if ext != "t1ce.nii.gz":
      self.error_msg.visible = True
      self.error_msg.text = "Upload the correct T1-ce file"
    else:
      self.error_msg.visible = False

  def t2_change(self, file, **event_args):
    """This method is called when a new file is loaded into this FileLoader"""
    ext = file.name.split("_")[-1]
    if ext != "t2.nii.gz":
      self.error_msg.visible = True
      self.error_msg.text = "Upload the correct T2 file"
    else:
      self.error_msg.visible = False
        

  def sliceno_lost_focus(self, **event_args):
    """This method is called when the TextBox loses focus"""
    if int(self.sliceno.text) not in range(30, 120):
      self.error_msg.visible = True
      self.error_msg.text = "Slice no should be between 30 and 119"
    else:
      self.error_msg.visible = False 

    

  def submit_click(self, **event_args):
    """This method is called when the button is clicked"""
    if self.error_msg.visible == True:
      self.error_msg.text = "The errors haven't been resolved"
    elif self.flair.file and self.t1.file and self.t1ce.file and self.t2.file and self.sliceno.text:
      ret_data = anvil.server.call('preprocess_predict', self.flair.file, self.t1.file, 
                                  self.t1ce.file, self.t2.file, self.sliceno.text)
      if ret_data:
        self.output.visible = True
        self.label_10.text="Slice no. {} of all 4 modalities".format(self.sliceno.text)
        self.flair_img.source = ret_data['flair_img']
        self.t1_img.source = ret_data['t1_img']
        self.t1ce_img.source = ret_data['t1ce_img']
        self.t2_img.source = ret_data['t2_img']
        
        self.unet_img.source = ret_data['unet_img']
        self.sobel_unet_img.source = ret_data['sobel_unet_img']
        self.vnet_img.source = ret_data['vnet_img']
        self.wnet_img.source = ret_data['wnet_img']      

    else:
      self.error_msg.visible = True
      self.error_msg.text = "Please fill all required details!"

      

  def link_6_click(self, **event_args):
    """This method is called when the link is clicked"""
    open_form('Form2')

  def link_7_click(self, **event_args):
    """This method is called when the link is clicked"""
    open_form('Form2')




















