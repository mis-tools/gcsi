#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkBinaryThresholdImageFilter.h"

#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "itkConstShapedNeighborhoodIterator.h"
#include "itkImageRegionConstIterator.h"

#include "itkLabelImageToLabelMapFilter.h"

#include "itkGradientMagnitudeImageFilter.h"

#include "itkNormalizeImageFilter.h"

#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryBallStructuringElement.h"

#include <stdio.h>
#include <map>
#include "limits.h"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>

#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>

using namespace std;
using namespace boost;

typedef float FloatPixelType;
typedef float MRPixelType;
typedef unsigned char BinaryLabelType;

typedef itk::Image<FloatPixelType, 3> FloatImageType;
typedef itk::Image<MRPixelType, 3> MRImageType;
typedef itk::Image<BinaryLabelType, 3> BinaryLabelImageType;

typedef itk::ImageFileReader<MRImageType> MRReaderType;
typedef itk::ImageFileReader<FloatImageType> FloatReaderType;

// Define graph traits and types
typedef float EdgeWeightType;
// typedef long NodeIndexType;
typedef int NodeIndexType;

typedef adjacency_list_traits < vecS, vecS, directedS > Traits;
typedef adjacency_list < vecS, vecS, directedS,
  property < vertex_name_t, std::string,
    property < vertex_index_t, NodeIndexType,
  property < vertex_color_t, boost::default_color_type,
         property < vertex_distance_t, NodeIndexType,
  property < vertex_predecessor_t, Traits::edge_descriptor > > > > >,
  property < edge_capacity_t, EdgeWeightType,
    property < edge_residual_capacity_t, EdgeWeightType,
      property < edge_reverse_t, Traits::edge_descriptor > > > > Graph;

FloatImageType::Pointer NormalizeImage(FloatImageType::Pointer spInputImage)
{
  typedef itk::NormalizeImageFilter< FloatImageType, FloatImageType > NormalizeFilterType;
  NormalizeFilterType::Pointer spNormalizeFilter = NormalizeFilterType::New();
  spNormalizeFilter->SetInput(spInputImage);
  spNormalizeFilter->Update();

  return spNormalizeFilter->GetOutput();
}

FloatImageType::Pointer ComputeContourPropabilisticMap(MRImageType::Pointer spInputImage)
{
  typedef itk::GradientMagnitudeImageFilter< MRImageType, FloatImageType > GradientMagnitudeFilterType;

  // Create and setup a gradient filter
  GradientMagnitudeFilterType::Pointer spGradientFilter = GradientMagnitudeFilterType::New();
  spGradientFilter->SetInput(spInputImage);
  spGradientFilter->Update();

  FloatImageType::Pointer spNormalizedImage = NormalizeImage(spGradientFilter->GetOutput());

  itk::ImageRegionIteratorWithIndex<FloatImageType> kIt(spNormalizedImage, spNormalizedImage->GetLargestPossibleRegion());

  while (!kIt.IsAtEnd())
  {
    FloatImageType::PixelType fNormalizedGradientValue = kIt.Get();

    kIt.Set(1.0f - exp(-fNormalizedGradientValue));

    ++kIt;
  }

  return spNormalizedImage;
}

template< typename ImgType >
typename ImgType::RegionType computeSegmentBoundingBox(itk::ImageRegionConstIteratorWithIndex<ImgType> imageIterator, int& iNumberOfVerticesToBeCreated, float fThreshold = 0.001)
{
  iNumberOfVerticesToBeCreated = 0;

  typename ImgType::IndexType::IndexValueType iMinX = std::numeric_limits<int>::max(), iMinY = std::numeric_limits<int>::max(), iMinZ = std::numeric_limits<int>::max();
  typename ImgType::IndexType::IndexValueType iMaxX = 0, iMaxY = 0, iMaxZ = 0;

  while (!imageIterator.IsAtEnd())
  {
    typename ImgType::IndexType kIndex3D = imageIterator.GetIndex();

    if (imageIterator.Get() > fThreshold)
    {
      iMinX = std::min(iMinX, kIndex3D[0]);
      iMinY = std::min(iMinY, kIndex3D[1]);
      iMinZ = std::min(iMinZ, kIndex3D[2]);

      iMaxX = std::max(iMaxX, kIndex3D[0]);
      iMaxY = std::max(iMaxY, kIndex3D[1]);
      iMaxZ = std::max(iMaxZ, kIndex3D[2]);

      iNumberOfVerticesToBeCreated++;
    }

    ++imageIterator;
  }

  typename ImgType::IndexType kIndex = {{iMinX, iMinY, iMinZ}};
  typename ImgType::SizeType kSize = {{iMaxX - iMinX, iMaxY - iMinY, iMaxZ - iMinZ}};
  typename ImgType::RegionType kBoundingBox(kIndex, kSize);
  return kBoundingBox;
}

// TODO: why can this not be set to a value between 1-254 e.g. 1 ?
const BinaryLabelType binaryMaskLabel = 255;

// Generate a mapping between the voxels inside bounding box of the segment and a vertex in the graph
int* GenerateIndexMap(BinaryLabelImageType::Pointer spSegmentROIMap, MRImageType::RegionType kBoundingBox)
{
  int iCurrentGraphCutVertex = 0;

  MRImageType::SizeType kSegmentDims = kBoundingBox.GetSize();
  MRImageType::IndexType kSegmentIndex = kBoundingBox.GetIndex();

  int* piIndexMapValues = (int*) malloc(kSegmentDims[0] * kSegmentDims[1] * kSegmentDims[2] * sizeof(int));

  itk::ImageRegionConstIteratorWithIndex<BinaryLabelImageType> imageIterator(spSegmentROIMap, kBoundingBox);

  while (!imageIterator.IsAtEnd())
  {
    BinaryLabelImageType::IndexType kIndex3D = imageIterator.GetIndex();

    int iIndexInIndexMap = (kIndex3D[0] - kSegmentIndex[0]) + (kIndex3D[1] - kSegmentIndex[1]) * kSegmentDims[0] + (kIndex3D[2] - kSegmentIndex[2]) * kSegmentDims[0] * kSegmentDims[1];

    if (imageIterator.Get() == binaryMaskLabel)
    {
      piIndexMapValues[iIndexInIndexMap] = iCurrentGraphCutVertex;
      iCurrentGraphCutVertex++;
    }
    else
    {
      piIndexMapValues[iIndexInIndexMap] = -1;
    }
    ++imageIterator;
  }

  return piIndexMapValues;
}

BinaryLabelImageType::Pointer ComputeBinaryInterestLabelMap(FloatImageType::Pointer kSegmentProbabilityMap, float fProbabilityThreshold, int iDilateRadius)
{
  typedef itk::BinaryThresholdImageFilter<FloatImageType, BinaryLabelImageType> BinaryThresholdImageFilter;
  BinaryThresholdImageFilter::Pointer spThresholdFilter = BinaryThresholdImageFilter::New();
  spThresholdFilter->SetInput(kSegmentProbabilityMap);
  spThresholdFilter->SetUpperThreshold(std::numeric_limits<FloatPixelType>::max()); // previously: 10
  spThresholdFilter->SetLowerThreshold(fProbabilityThreshold);
  spThresholdFilter->SetInsideValue(binaryMaskLabel);
  spThresholdFilter->SetOutsideValue(0);
  spThresholdFilter->Update();
  // BinaryLabelImageType::Pointer spBinaryImage = spThresholdFilter->GetOutput();
  // return spBinaryImage;

  typedef itk::BinaryBallStructuringElement<BinaryLabelImageType::PixelType, BinaryLabelImageType::ImageDimension> StructuringElementType;
  StructuringElementType structuringElement;
  structuringElement.SetRadius(iDilateRadius);
  structuringElement.CreateStructuringElement();

  typedef itk::BinaryDilateImageFilter <BinaryLabelImageType, BinaryLabelImageType, StructuringElementType> BinaryDilateImageFilterType;

  BinaryDilateImageFilterType::Pointer spDilateFilter = BinaryDilateImageFilterType::New();
  spDilateFilter->SetInput(spThresholdFilter->GetOutput());
  spDilateFilter->SetKernel(structuringElement);
  spDilateFilter->Update();

  BinaryLabelImageType::Pointer spDilatedImage = spDilateFilter->GetOutput();

  return spDilatedImage;
}

Traits::edge_descriptor AddEdge(Traits::vertex_descriptor &v1, Traits::vertex_descriptor &v2, property_map < Graph, edge_reverse_t >::type &rev, const double capacity, Graph &g)
{
  Traits::edge_descriptor e1 = add_edge(v1, v2, g).first;
  Traits::edge_descriptor e2 = add_edge(v2, v1, g).first;
  put(edge_capacity, g, e1, capacity);
  put(edge_capacity, g, e2, capacity);

  rev[e1] = e2;
  rev[e2] = e1;
  return e1;
}

template <class T>
bool contains(std::vector<T> const &v, T const &x)
{
  return !(v.empty() || std::find(v.begin(), v.end(), x) == v.end());
}

template <typename LabelPixelType> int GraphCutSegmentationImprover();

float fSigma = 1.0f; // is the robust scale of image (article does not suggest value)
float fLambda = 2.0f; // weighting between the data and the smoothness term.
float fC = 0.5f; // weighting between intensity term and boundary term.
float fAlpha = 0.2f; // weighting between tissue type and segment probability information
float fThreshold = 0.001;
float fSegmentProbabilityTreshold = 0; // if 0 the narrow band algorithm is not used

unsigned int iInsideValue = 1;
unsigned int iOutsideValue = 0;
unsigned int iUpperValue = std::numeric_limits<unsigned int>::max();
unsigned int iLowerValue = 1;
unsigned int numberOfBitsPerVoxel = 0; // 8=>unsigned char, 32=>int
unsigned int verboseOutput = 0; // 0=false, 1=true

std::vector<unsigned int> labelsToDrop;
std::map<int, std::string> graphCutCleaningSegments;

std::string mrImageFilePath = "";
std::string labelMapImageFilePath = "";
std::string outputLabelMapImageFilePath = "";
std::vector<std::string> foregroundFiles;
std::vector<std::string> backgroundFiles;
std::string contourPropabilisticMapFilename = "";
std::string spBinaryROIMapFilename = "";

int process(int argc, char* argv[])
{
  int iC;
  int iSegmentID;
  std::string filename;
  while ((iC = getopt(argc, argv, "G:N:V:B:I:O:L:U:C:a:s:l:c:p:f:b:t:d:n:")) != -1)
  {
    std::stringstream ss(optarg);
    switch (iC)
    {
    case 'V':
      ss >> verboseOutput;
      break;
    case 'G': // normalized gradient image
      ss >> contourPropabilisticMapFilename;
      break;
    case 'N': // narrow band binary ROI
      ss >> spBinaryROIMapFilename;
      break;
    case 'B':
      ss >> numberOfBitsPerVoxel;
      break;
    case 'I':
      ss >> iInsideValue;
      break;
    case 'O':
      ss >> iOutsideValue;
      break;
    case 'L':
      ss >> iLowerValue;
      break;
    case 'U':
      ss >> iUpperValue;
      break;
    case 'a':
      ss >> fAlpha;
      break;
    case 's':
      ss >> fSigma;
      break;
    case 'l':
      ss >> fLambda;
      break;
    case 't':
      ss >> fThreshold;
      break;
    case 'n':
      ss >> fSegmentProbabilityTreshold;
      break;
    case 'C':
      ss >> fC;
      break;
    case 'c':
      ss >> iSegmentID;
      graphCutCleaningSegments[iSegmentID] = "";
      break;
    case 'p':
      ss >> filename;
      graphCutCleaningSegments[iSegmentID] = filename;
      break;
    case 'f':
      ss >> filename;
      foregroundFiles.push_back(filename);
      break;
    case 'b':
      ss >> filename;
      backgroundFiles.push_back(filename);
      break;
    case 'd':
      unsigned int segmentToDrop;
      ss >> segmentToDrop;
      if (!contains<unsigned int>(labelsToDrop, segmentToDrop))
        labelsToDrop.push_back(segmentToDrop);
      break;
    default:
      return EXIT_FAILURE;
    }
  }

  mrImageFilePath = argv[optind];
  labelMapImageFilePath = argv[optind + 1];
  outputLabelMapImageFilePath = argv[optind + 2];

  bool printArguments = verboseOutput > 0;
  if (printArguments)
  {
    std::cout << "----------------" << std::endl;
    std::cout << "input parameters" << std::endl;
    std::cout << "----------------" << std::endl;

    std::cout << "mrImageFilePath: " << mrImageFilePath << std::endl;
    std::cout << "labelMapImageFilePath: " << labelMapImageFilePath << std::endl;
    std::cout << "outputLabelMapImageFilePath: " << outputLabelMapImageFilePath << std::endl;
    std::cout << "inside: " << iInsideValue << ", outside: " << iOutsideValue;
    std::cout << ", lower: " << iLowerValue << ", upper: " << iUpperValue << std::endl;
    std::cout << "alpha: " << fAlpha << ", sigma: " << fSigma << ", C: " << fC;
    std::cout << ", lambda: " << fLambda << ", threshold: " << fThreshold << std::endl;
    std::cout << "narrowBandProbabilityThreshold: " << fSegmentProbabilityTreshold << std::endl;

    std::cout << "labelsToDrop: [";
    bool first_itr = true;
    for (std::vector<unsigned int>::iterator it = labelsToDrop.begin(); it != labelsToDrop.end(); it++)
    {
      if (!first_itr)
        std::cout << ",";
      std::cout << *it;
      first_itr = false;
    }
    std::cout << "]" << std::endl;

    std::cout << "graphCutCleaningSegments: [";
    first_itr = true;
    std::map<int, std::string>::iterator it;
    for (it = graphCutCleaningSegments.begin(); it != graphCutCleaningSegments.end(); it++)
    {
      if (!first_itr)
        std::cout << ",";
      std::cout << it->first << ':' << it->second;
      first_itr = false;
    }
    std::cout << "]" << std::endl;

    std::cout << "foregroundFiles: [";
    first_itr = true;
    for (std::vector<std::string>::iterator it = foregroundFiles.begin(); it != foregroundFiles.end(); ++it)
    {
      if (!first_itr)
        std::cout << ",";
      std::cout << *it;
      first_itr = false;
    }
    std::cout << "]" << std::endl;

    std::cout << "backgroundFiles: [";
    first_itr = true;
    for (std::vector<std::string>::iterator it = backgroundFiles.begin(); it != backgroundFiles.end(); ++it)
    {
      if (!first_itr)
        std::cout << ",";
      std::cout << *it;
      first_itr = false;
    }
    std::cout << "]" << std::endl;
    std::cout << "----------------" << std::endl;
  } // end if(printArguments)

  switch (numberOfBitsPerVoxel)
  {
  case 8:
    return GraphCutSegmentationImprover<unsigned char>();
    break;
  case 16:
    return GraphCutSegmentationImprover<unsigned short>();
    break;
  case 32:
    return GraphCutSegmentationImprover<int>();
    break;
  default:
    std::cerr << "unknown numberOfBitsPerVoxel: ";
    std::cerr << numberOfBitsPerVoxel << std::endl;
    return EXIT_FAILURE;
  }
}

template <typename LabelPixelType>
int GraphCutSegmentationImprover()
{
  typedef itk::Image<LabelPixelType, 3> LabelImageType;
  MRReaderType::Pointer reader1 = MRReaderType::New();
  reader1->SetFileName(mrImageFilePath);
  typedef itk::ImageFileReader<LabelImageType> LabelReaderType;
  typename LabelReaderType::Pointer reader2 = LabelReaderType::New();
  reader2->SetFileName(labelMapImageFilePath);

  std::vector<FloatImageType::Pointer> foregroundImgs;
  std::vector<FloatImageType::Pointer> backgroundImgs;

  try
  {
    reader1->Update();
    reader2->Update();

    for (std::vector<std::string>::iterator it = foregroundFiles.begin(); it != foregroundFiles.end(); ++it)
    {
      FloatReaderType::Pointer reader = FloatReaderType::New();
      reader->SetFileName(*it);
      reader->Update();
      FloatImageType::Pointer foregroundImg = reader->GetOutput();
      foregroundImgs.push_back(foregroundImg);
    }

    for (std::vector<std::string>::iterator it = backgroundFiles.begin(); it != backgroundFiles.end(); ++it)
    {
      FloatReaderType::Pointer reader = FloatReaderType::New();
      reader->SetFileName(*it);
      reader->Update();
      FloatImageType::Pointer backgroundImg = reader->GetOutput();
      backgroundImgs.push_back(backgroundImg);
    }
  }
  catch (itk::ExceptionObject& kExcp)
  {
    std::cerr << kExcp << std::endl;
    return EXIT_FAILURE;
  }

  MRImageType::Pointer spMRImage = NormalizeImage(reader1->GetOutput());
  typename LabelImageType::Pointer spLabelMapImage = reader2->GetOutput();
  FloatImageType::Pointer spContourPropabilisticMap = ComputeContourPropabilisticMap(spMRImage);

  if (contourPropabilisticMapFilename != "")
  {
    if (verboseOutput)
    {
      std::cout << "Writing contourPropabilisticMap to: ";
      std::cout << contourPropabilisticMapFilename << std::endl;
    }

    typedef itk::ImageFileWriter<MRImageType> MRWriterType;
    MRWriterType::Pointer writer = MRWriterType::New();
    writer->SetFileName(contourPropabilisticMapFilename);
    writer->SetInput(spContourPropabilisticMap);
    writer->SetUseCompression(1);
    try
    {
      writer->Update();
    }
    catch (itk::ExceptionObject& exp)
    {
      std::cout << exp << std::endl;
    }
  }

  MRImageType::SizeType kDims = spMRImage->GetLargestPossibleRegion().GetSize();

  typedef itk::BinaryThresholdImageFilter<LabelImageType, LabelImageType> BinaryThresholdImageFilter;
  typename BinaryThresholdImageFilter::Pointer spThresholdFilter = BinaryThresholdImageFilter::New();
  spThresholdFilter->SetInput(spLabelMapImage);
  spThresholdFilter->SetUpperThreshold(iUpperValue);
  spThresholdFilter->SetLowerThreshold(iLowerValue);
  spThresholdFilter->SetInsideValue(iInsideValue);
  spThresholdFilter->SetOutsideValue(iOutsideValue);
  spThresholdFilter->Update();
  typename LabelImageType::Pointer spOutputImage = spThresholdFilter->GetOutput();

  typedef itk::LabelImageToLabelMapFilter<LabelImageType> LabelImageToLabelMapFilterType;
  typename LabelImageToLabelMapFilterType::Pointer spLabelImageToLabelMapFilter = LabelImageToLabelMapFilterType::New();
  spLabelImageToLabelMapFilter->SetInput(spLabelMapImage);
  spLabelImageToLabelMapFilter->Update();

  for (unsigned int i = 0; i < spLabelImageToLabelMapFilter->GetOutput()->GetNumberOfLabelObjects(); ++i)
  {
    typename LabelImageToLabelMapFilterType::OutputImageType::LabelObjectType* pkLabelObject = spLabelImageToLabelMapFilter->GetOutput()->GetNthLabelObject(i);

    const int iLabel = pkLabelObject->GetLabel();

    if (contains<unsigned int>(labelsToDrop, iLabel))
    {
      std::cout << "Dropping label: " << iLabel << std::endl;
      continue;
    }

    if (graphCutCleaningSegments.find(iLabel) != graphCutCleaningSegments.end())
    {
      std::cout << "Processing label: " << iLabel << std::endl;
      FloatReaderType::Pointer reader = FloatReaderType::New();
      reader->SetFileName(graphCutCleaningSegments[iLabel]);
      try
      {
        reader->Update();
      }
      catch (itk::ExceptionObject& kExcp)
      {
        std::cerr << kExcp << std::endl;
        return EXIT_FAILURE;
      }

      FloatImageType::Pointer spSegmentPropImage = reader->GetOutput();

      // Number of potential "inside" vertices in the image graph,
      // 2 more vertices are added (source and sink)
      int iNumberOfVertices;

      // Compute tight bounding box position and dim, and also
      // fill out the number of internal vertices needed
      MRImageType::RegionType kSegmentBoundingBox;
      MRImageType::SizeType kSegmentDims;

      int* piVoxelToVertexIndexMappingData = NULL;
      bool useNarrowBandOptimization = fSegmentProbabilityTreshold > 0;
      if (useNarrowBandOptimization)
      {
        std::cout << "using narrow band algorithm, with: " << fSegmentProbabilityTreshold << std::endl;
        int iDilateRadius = 1;
        BinaryLabelImageType::Pointer spBinaryROIMap = ComputeBinaryInterestLabelMap(spSegmentPropImage, fSegmentProbabilityTreshold, iDilateRadius);

        if (spBinaryROIMapFilename != "")
        {
          if (verboseOutput)
          {
            std::cout << "Writing spBinaryROIMap to: ";
            std::cout << spBinaryROIMapFilename << std::endl;
          }
          typedef itk::ImageFileWriter<BinaryLabelImageType> BinaryLabelImageWriterType;
          BinaryLabelImageWriterType::Pointer writer = BinaryLabelImageWriterType::New();
          writer->SetFileName(spBinaryROIMapFilename);
          writer->SetInput(spBinaryROIMap);
          writer->SetUseCompression(1);
          try
          {
            writer->Update();
          }
          catch (itk::ExceptionObject& exp)
          {
            std::cout << exp << std::endl;
          }
        }

        itk::ImageRegionConstIteratorWithIndex<BinaryLabelImageType> imageIterator(spBinaryROIMap, spBinaryROIMap->GetLargestPossibleRegion());

        kSegmentBoundingBox = computeSegmentBoundingBox(imageIterator, iNumberOfVertices, fThreshold);

        piVoxelToVertexIndexMappingData = GenerateIndexMap(spBinaryROIMap, kSegmentBoundingBox);
        std::cout << "Voxel to vertex index mapping data generated " << std::endl;

        kSegmentDims = kSegmentBoundingBox.GetSize();
      }
      else
      {
        std::cout << "using standard algorithm" << std::endl;
        itk::ImageRegionConstIteratorWithIndex<MRImageType> imageIterator(spSegmentPropImage, spSegmentPropImage->GetLargestPossibleRegion());

        kSegmentBoundingBox = computeSegmentBoundingBox(imageIterator, iNumberOfVertices, fThreshold);

        kSegmentDims = kSegmentBoundingBox.GetSize();
        iNumberOfVertices = kSegmentDims[0] * kSegmentDims[1] * kSegmentDims[2];
      }
      std::cout << "Segment bounding box : " << kSegmentBoundingBox << std::endl;
      std::cout << "Number of vertices required : " << iNumberOfVertices << "+2" << std::endl;

      // The graph initialized with some nodes
      Graph g(iNumberOfVertices + 2);
      unsigned int uiSourceIndex = iNumberOfVertices;
      unsigned int uiSinkIndex = uiSourceIndex + 1;
      Traits::vertex_descriptor kSourceNode = vertex(uiSourceIndex, g);
      Traits::vertex_descriptor kSinkNode = vertex(uiSinkIndex, g);

      // Reverse edge property map
      property_map < Graph, edge_reverse_t >::type rev = get(edge_reverse, g);

      // Color property map
      property_map<Graph, vertex_color_t>::type sourceOrSink = get(vertex_color, g);

      // Set up t-weights
      {
        itk::ImageRegionConstIteratorWithIndex<FloatImageType> imageIterator(spSegmentPropImage, kSegmentBoundingBox);

        for (; !imageIterator.IsAtEnd(); ++imageIterator)
        {
          MRImageType::IndexType kIndex3D = imageIterator.GetIndex();
          MRImageType::OffsetType kOffSetIndex3D = kIndex3D - kSegmentBoundingBox.GetIndex();

          int iIndex1D = kOffSetIndex3D[2] * kSegmentDims[0] * kSegmentDims[1] + kOffSetIndex3D[1] * kSegmentDims[0] + kOffSetIndex3D[0];

          if (useNarrowBandOptimization)
          {
            int iMappedIndex1D = piVoxelToVertexIndexMappingData[iIndex1D];
            // Vertex id is -1 if voxel is not in narrow band binary mask
            if (iMappedIndex1D < 0)
              continue;
          }

          float fTissueForegroundProbability = 0;
          for (std::vector<FloatImageType::Pointer>::iterator it = foregroundImgs.begin(); it != foregroundImgs.end(); ++it)
          {
            FloatImageType::Pointer foregroundImgs = *it;
            fTissueForegroundProbability += std::max(0.001f, foregroundImgs->GetPixel(kIndex3D));
          }
          if (foregroundImgs.size() > 0)
            fTissueForegroundProbability /= foregroundImgs.size();

          float fTissueBackgroundProbability = 0;
          for (std::vector<FloatImageType::Pointer>::iterator it = backgroundImgs.begin(); it != backgroundImgs.end(); ++it)
          {
            FloatImageType::Pointer backgroundImgs = *it;
            fTissueBackgroundProbability += std::max(0.001f, backgroundImgs->GetPixel(kIndex3D));
          }
          if (backgroundImgs.size() > 0)
            fTissueBackgroundProbability /= backgroundImgs.size();

          float fSegmentForegroundProbability = std::max(0.001f, imageIterator.Get());
          float fSegmentBackgroundProbability = std::max(0.001f, 1.0f - fSegmentForegroundProbability);

          float fRegionBackground = -fAlpha * log(fTissueBackgroundProbability) - (1.0f - fAlpha) * log(fSegmentBackgroundProbability);
          float fRegionObject = -fAlpha * log(fTissueForegroundProbability) - (1.0f - fAlpha) * log(fSegmentForegroundProbability);

          Traits::vertex_descriptor kNode;
          if (useNarrowBandOptimization)
          {
            int iMappedIndex1D = piVoxelToVertexIndexMappingData[iIndex1D];
            kNode = vertex(iMappedIndex1D, g);
          }
          else
          {
            kNode = vertex(iIndex1D, g);
          }

          // Add source edge to index with weight fLambda*fRegionBackground and
          // sink edge from index with weight fLambda*fRegionObject
          AddEdge(kSourceNode, kNode, rev, fLambda * fRegionBackground, g);
          AddEdge(kNode, kSinkNode, rev, fLambda * fRegionObject, g);
        } // loop
      }

      std::cout << "Done setting up t-weights" << std::endl;

      // Set up n-weights
      typedef itk::ConstShapedNeighborhoodIterator<MRImageType> IteratorType;

      itk::Size<3> radius;
      radius.Fill(1);
      IteratorType iterator(radius, spMRImage, kSegmentBoundingBox);

      iterator.ClearActiveList();

      IteratorType::OffsetType pzz = {{1, 0, 0}};
      iterator.ActivateOffset(pzz);
      IteratorType::OffsetType zpz = {{0, 1, 0}};
      iterator.ActivateOffset(zpz);
      IteratorType::OffsetType zzp = {{0, 0, 1}};
      iterator.ActivateOffset(zzp);

      // Add the negative half of the stencil as it is needed to compute fK
      IteratorType::OffsetType nzz = {{-1, 0, 0}};
      iterator.ActivateOffset(nzz);
      IteratorType::OffsetType znz = {{0, -1, 0}};
      iterator.ActivateOffset(znz);
      IteratorType::OffsetType zzn = {{0, 0, -1}};
      iterator.ActivateOffset(zzn);

      // float fK = 0.0f;
      for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
      {
        // float fKTemp = 1.0f;

        IteratorType::ConstIterator ci = iterator.Begin();
        MRImageType::IndexType kIndexHere3D = iterator.GetIndex();

        MRImageType::OffsetType kOffSetIndex3D = kIndexHere3D - kSegmentBoundingBox.GetIndex();
        int iIndexHere1D = kOffSetIndex3D[2] * kSegmentDims[0] * kSegmentDims[1] + kOffSetIndex3D[1] * kSegmentDims[0] + kOffSetIndex3D[0];

        if (useNarrowBandOptimization)
        {
          int iMappedIndexHere1D = piVoxelToVertexIndexMappingData[iIndexHere1D];
          // Vertex id is -1 if voxel is not in narrow band binary mask
          if (iMappedIndexHere1D < 0)
            continue;
        }

        float fIntensityHere = iterator.GetCenterPixel();

        float fContourPropabilityHere = spContourPropabilisticMap->GetPixel(kIndexHere3D);

        for (; !ci.IsAtEnd(); ci++)
        {
          MRImageType::OffsetType kOffset3D = ci.GetNeighborhoodOffset();
          MRImageType::IndexType kIndexThere3D = iterator.GetIndex() + kOffset3D;

          if (kSegmentBoundingBox.IsInside(kIndexThere3D))
          {
            MRImageType::OffsetType kOffsetIndexThere3D = kIndexThere3D - kSegmentBoundingBox.GetIndex();

            int iIndexThere1D = kOffsetIndexThere3D[2] * kSegmentDims[0] * kSegmentDims[1] + kOffsetIndexThere3D[1] * kSegmentDims[0] + kOffsetIndexThere3D[0];

            if (useNarrowBandOptimization)
            {
              int iMappedIndexThere1D = piVoxelToVertexIndexMappingData[iIndexThere1D];
              // Vertex id is -1 if voxel is not in narrow band binary mask
              if (iMappedIndexThere1D < 0)
                continue;
            }

            float fIntensityThere = ci.Get();

            // float fInvSigmaS = 1.0f / (2.0f * fSigma * fSigma);
            // float fIntensityTerm = exp(-pow(fIntensityHere - fIntensityThere, 2.0f) * fInvSigmaS);

            float fIntensityTerm = 1.0f / (1.0f + log(1.0f + 0.5f * pow((fIntensityHere - fIntensityThere) / fSigma, 2.0f)));

            float fAtoBIntensityTerm = 1.0f;
            float fBtoAIntensityTerm = 1.0f;

            if (fIntensityHere > fIntensityThere)
            {
              fAtoBIntensityTerm = fIntensityTerm;
            }
            else
            {
              fBtoAIntensityTerm = fIntensityTerm;
            }

            float fContourPropabilityThere = spContourPropabilisticMap->GetPixel(kIndexThere3D);

            // float fLargestContourPropability = std::max(fContourPropabilityHere, fContourPropabilityThere);
            // float fEdgeTerm = 1.0f - fLargestContourPropability;

            float fAtoBEdgeTerm = 1.0f;
            float fBtoAEdgeTerm = 1.0f;

            if (fContourPropabilityHere > fContourPropabilityThere)
            {
              fAtoBEdgeTerm = 1.0f - fContourPropabilityHere;
            }
            else
            {
              fBtoAEdgeTerm = 1.0f - fContourPropabilityThere;
            }

            float fAtoBTerm = fC * fAtoBIntensityTerm + (1.0f - fC) * fAtoBEdgeTerm;
            float fBtoATerm = fC * fBtoAIntensityTerm + (1.0f - fC) * fBtoAEdgeTerm;

            // fKTemp += fAtoBTerm;

            // Only add edge for positive half of stencil
            if (kOffset3D[0] >= 0 && kOffset3D[1] >= 0 && kOffset3D[2] >= 0)
            {
              Traits::edge_descriptor e1, e2;
              if (useNarrowBandOptimization)
              {
                int iMappedIndexHere1D = piVoxelToVertexIndexMappingData[iIndexHere1D];
                int iMappedIndexThere1D = piVoxelToVertexIndexMappingData[iIndexThere1D];
                e1 = add_edge(iMappedIndexHere1D, iMappedIndexThere1D, g).first;
                e2 = add_edge(iMappedIndexThere1D, iMappedIndexHere1D, g).first;
              }
              else
              {
                e1 = add_edge(iIndexHere1D, iIndexThere1D, g).first;
                e2 = add_edge(iIndexThere1D, iIndexHere1D, g).first;
              }

              put(edge_capacity, g, e1, fAtoBTerm);
              put(edge_capacity, g, e2, fBtoATerm);
              rev[e1] = e2;
              rev[e2] = e1;
            }
          }
        } // loop

        // fK = std::max(fK, fKTemp);
      }

      std::cout << "Done setting up n-weights" << std::endl;

      EdgeWeightType flow = boykov_kolmogorov_max_flow(g, kSourceNode, kSinkNode);
      std::cout << "Found a flow of: " << flow << " for segment " << iLabel << std::endl;
      {
        itk::ImageRegionIteratorWithIndex<LabelImageType> imageIterator(spOutputImage, kSegmentBoundingBox);

        for (; !imageIterator.IsAtEnd(); ++imageIterator)
        {
          typename LabelImageType::IndexType kIndex3D = imageIterator.GetIndex();
          MRImageType::OffsetType kOffSetIndex3D = kIndex3D - kSegmentBoundingBox.GetIndex();

          int iIndex1D = kOffSetIndex3D[2] * kSegmentDims[0] * kSegmentDims[1] + kOffSetIndex3D[1] * kSegmentDims[0] + kOffSetIndex3D[0];

          if (useNarrowBandOptimization)
          {
            int iMappedIndex1D = piVoxelToVertexIndexMappingData[iIndex1D];
            // Vertex id is -1 if voxel is not in narrow band binary mask
            if (iMappedIndex1D < 0)
              continue;

            if (sourceOrSink[iMappedIndex1D] != sourceOrSink[uiSinkIndex])
              imageIterator.Set(iLabel);
          }
          else
          {
            if (sourceOrSink[iIndex1D] != sourceOrSink[uiSinkIndex])
              imageIterator.Set(iLabel);
          }
        } // loop
      }
      if (useNarrowBandOptimization)
        free(piVoxelToVertexIndexMappingData);
    } // fi: iLabel is in graphCutCleaningSegments
    else
    {
      // Copy label over
      std::cout << "Copying label: " << iLabel << std::endl;
      itk::ImageRegionConstIterator<LabelImageType> it(spLabelMapImage, spLabelMapImage->GetRequestedRegion());
      itk::ImageRegionIterator<LabelImageType> it2(spOutputImage, spOutputImage->GetRequestedRegion());
      while (!it.IsAtEnd())
      {
        if (it.Get() == iLabel)
        {
          it2.Set(iLabel);
        }
        ++it2;
        ++it;
      }
    }
  }

  typedef itk::ImageFileWriter<LabelImageType> LabelWriterType;
  typename LabelWriterType::Pointer writer = LabelWriterType::New();
  writer->SetFileName(outputLabelMapImageFilePath);
  writer->SetInput(spOutputImage);
  writer->SetUseCompression(1);
  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject& exp)
  {
    std::cout << exp << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cout << "Usage: " << argv[0];
    std::cout << " mrImage";
    std::cout << " labelMapImage";
    std::cout << " outputLabelMapImage";
    std::cout << " [-a alpha] [-s sigma] [-l lambda] [-t threshold]";
    std::cout << " [-n narrowBandProbabilityThreshold]";
    std::cout << " [-I InsideValue] [-O OutsideValue]";
    std::cout << " [-L LowerValue] [-U UpperValue]";
    std::cout << " [-c graph_cut_clean_segmentID -p segmentProbabilityMapImage]";
    std::cout << " [-f foregroundProbabilityMapImage]";
    std::cout << " [-b backgroundProbabilityMapImage]";
    std::cout << " [-G normalized gradient image/contourPropabilisticMap]";
    std::cout << " [-N narrowband binary ROI/spBinaryROIMap (only output if narrowband is used)]";
    std::cout << " [-V verboseOutput (0)/1]";
    std::cout << std::endl;
    if (argc >= 2 &&
        (std::string(argv[1]) == std::string("--help") ||
         std::string(argv[1]) == std::string("-h")))
    {
      return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
  }
  return process(argc, argv);
}
