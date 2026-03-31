terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  tags = {
    Project     = "seattle-energy-predictor"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ECR registry-level scanning configuration (replaces deprecated per-repo scan_on_push)
resource "aws_ecr_registry_scanning_configuration" "main" {
  scan_type = "BASIC"

  rule {
    scan_frequency = "SCAN_ON_PUSH"

    repository_filter {
      filter      = "*"
      filter_type = "WILDCARD"
    }
  }
}

# ECR repository to store the BentoML Docker image
resource "aws_ecr_repository" "seattle" {
  name                 = var.repo_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  tags = local.tags
}

# IAM role allowing App Runner to pull images from ECR
resource "aws_iam_role" "apprunner_ecr" {
  name = "AppRunnerECRAccessRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "build.apprunner.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}
