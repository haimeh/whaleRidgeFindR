#define cimg_display 0
// [[Rcpp::depends(imager)]]
#include <imager.h>
#include <vector>
#include <algorithm>


//' simplifyAngles
//' 
//' This function returns a cimg of type int.
//' Descritizes vectors represented by their angle into one of 4 sectors.
//' Best visualized has the right half of the unit circle devided in 4
//'
//' @param ang cimg reference of type double
cimg_library::CImg<int> simplifyAngles(const cimg_library::CImg<double> ang)
{
  cimg_library::CImg<int> simplifiedAngles(ang.width(),ang.height(),1,1,0);
  int approxAng = 0;
  const double pi = 3.1415926535897932;
  cimg_forXY(ang,x,y)
  {
         if (ang(x,y) <= ( 4*pi/8) && ang(x,y) > ( 2*pi/8)){approxAng = 1;}
    else if (ang(x,y) <= ( 2*pi/8) && ang(x,y) > ( 0.0   )){approxAng = 2;}
    else if (ang(x,y) <= ( 0.0   ) && ang(x,y) > (-2*pi/8)){approxAng = 3;}
    else {approxAng = 0;}
    
    simplifiedAngles(x,y) = approxAng;
  }
  return simplifiedAngles;
}

//' nonMaxSuppress
//' 
//' This function returns a cimg of type double.
//' Verifies the allignment between
//' The apex of the gradient magnitude of x and y in the 3x3 region around each pixel,
//' with the angle range estimated from the \code{simplifyAngles} function.
//' Allignment is soft, in that overlap is allowed.
//' 
//' @param edge cimg reference of type double
//' @param ang cimg reference of type int
cimg_library::CImg<double> nonMaxSuppress(const cimg_library::CImg<double>& edge, 
                    const cimg_library::CImg<int>& ang)
{
  cimg_library::CImg<double> newedge(edge.width(),edge.height() ,1,1,0);
  double targetPixel = 0.0;
  
  cimg_for_insideXY(edge,col,row,1)
  {
    targetPixel = edge(col,row);
    switch(ang(col,row))
    {
    case 1:
      if(((edge(col  ,row-1) < targetPixel) &&
          (edge(col  ,row+1) < targetPixel)) ||
         ((edge(col+1,row+1) < targetPixel) &&
          (edge(col-1,row-1) < targetPixel)))
        {newedge(col,row)=targetPixel;}
      break;
    
    case 2:
      if(((edge(col+1,row+1) < targetPixel) && 
          (edge(col-1,row-1) < targetPixel)) ||
         ((edge(col+1,row  ) < targetPixel) && 
          (edge(col-1,row  ) < targetPixel)))
        {newedge(col,row)=targetPixel;}
      break;
    
    case 3:
      if(((edge(col+1,row  ) < targetPixel) &&
          (edge(col-1,row  ) < targetPixel)) ||
         ((edge(col+1,row-1) < targetPixel) &&
          (edge(col-1,row+1) < targetPixel)))
        {newedge(col,row)=targetPixel;}
      break;
      
    case 0:
      if(((edge(col+1,row-1) < targetPixel) &&
          (edge(col-1,row+1) < targetPixel)) ||
         ((edge(col  ,row-1) < targetPixel) &&
          (edge(col  ,row+1) < targetPixel)))
        {newedge(col,row)=targetPixel;}
      break;
    }
  }
  return newedge;
}

//' extractEdgeMap
//' 
//' This function returns a cimg wrapped as NumeriMatrix.
//' using the gradient magnitude in the x and y directions,
//' estimated by the imager function \code{imgradient}
//' and the greadient angles
//' estimated as atan(dy/dx) and subsequently simplified and discretized
//' the canny edges are calculated, 
//' defined as the points in the gradient magnitude map that are 
//' greater than the 4(of the 8 posible) neighbors, orthogonal to the estimated angle
//'
//' @param gradientFromR cimg wraped as NumericVector
//' @param anglesFromR cimg wraped as NumericVector
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector extractEdgeMap(const Rcpp::NumericVector gradientFromR, 
                                   const Rcpp::NumericVector anglesFromR)
{
  cimg_library::CImg<double> gradient = Rcpp::as< cimg_library::CImg<double> >(gradientFromR);
  cimg_library::CImg<double> angles = Rcpp::as< cimg_library::CImg<double> >(anglesFromR);
  return Rcpp::wrap(nonMaxSuppress(gradient,simplifyAngles(angles)));
}
