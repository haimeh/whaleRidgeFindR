% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/extractFeatures_old.R
\name{traceFromImage}
\alias{traceFromImage}
\title{traceFromImage}
\usage{
traceFromImage(
  whaleRidge,
  startStopCoords = NULL,
  pathNet = NULL,
  justStartCoord = NULL
)
}
\arguments{
\item{startStopCoords}{list of 3 coordinates: leadingEnd, startPoint, trailingEnd. If NULL, these points are estimated}

\item{pathNet}{mxnet model for isolating trailing edge}

\item{fin}{Value of type cimg. Load the image via load.image("directory/finImage.JPG")}
}
\value{
Value of type list containing:
"coordinates" a dataframe of coordinates
"annulus" a 3 channel cimg of isolated features
"dim" a vector denoting the x and y dims the coordinates are for
}
\description{
traceFromImage
}
\details{
Processes an image(cimg) containing a fin. 
First the image undergoes cleanup through a variety of filters and glare removal via
\code{constrainSizeFinImage} and \code{fillGlare}
These processes help enhance edge clarity.
The trailing edge is highlighted via neural network. 
The image is then cropped down to the trailing edge for efficiency purposes.
The canny edges are then extracted from the crop and passed to 
\code{traceFromCannyEdges}
which isolates coordinates for the trailing edge. These coordinates are then passed to
\code{extractAnnulus}
which collects image data used for identification.
Both the coordinates and the image annulus are then returned.
}
