# *****************************************************************************
#  API ----
#
# Course Code: BBT4206
# Course Name: Business Intelligence II
# Semester Duration: 21st August 2023 to 28th November 2023
#
# Lecturer: Allan Omondi
# Contact: aomondi [at] strathmore.edu
#
# Note: The lecture contains both theory and practice. This file forms part of
#       the practice. It has required lab work submissions that are graded for
#       coursework marks.
#
# License: GNU GPL-3.0-or-later
# See LICENSE file for licensing information.
# *****************************************************************************

#command to list all the libraries available in your
# computer:
.libPaths()


# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)

# If renv::restore() did not install the "languageserver" package (required to
# use R for VS Code), then it can be installed manually as follows (restart R
# after executing the command):

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----

# We can create an API to access the model from outside R using a package
# called Plumber.

# STEP 1. Install and Load the Required Packages ----
## plumber ----
if (require("plumber")) {
  require("plumber")
} else {
  install.packages("plumber", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# This requires the "plumber" package that was installed and loaded earlier in
# STEP 1. The commenting below makes R recognize the code as the definition of
# an API, i.e., #* comments.

recommendation_caret_model_lda <- readRDS("./data/saved_recommendation_caret_model_lda.rds")

#* @apiTitle Crop Prediction Model API

#* @apiDescription Used to predict which crops can do well under certain conditions.

#* @param arg_nitrogen The amount of nitrogen in the soil
#* @param arg_phosphorous Amount of phosphorous in the soil
#* @param arg_pottasium Amount of potassium in the soil
#* @param arg_temperatures Temperatures in the atmosphere 
#* @param arg_humidity Amount of humidity in the atmosphere 
#* @param arg_pH Amount of pH in the soil
#* @param arg_rainfall Average amount of rainfall in the area 


#* @get /label

predict_crop <-
  function(arg_nitrogen, arg_phosphorous, arg_pottasium, arg_temperatures, arg_humidity,
           arg_pH, arg_rainfall) {
    # Create a data frame using the arguments
    to_be_predicted <-
      data.frame(nitrogen = as.numeric(arg_nitrogen),
                 phosphorous = as.numeric(arg_phosphorous),
                 pottasium = as.numeric(arg_pottasium),
                 temperatures = as.numeric(arg_temperatures),
                 humidity = as.numeric(arg_humidity),
                 pH = as.numeric(arg_pH),
                 rainfall = as.numeric(arg_rainfall))
            
    # Make a prediction based on the data frame
    predict(loaded_recommendation_model_lda, to_be_predicted)
  }

