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

# Read outputs from the infra module state — no manual var passing needed
data "terraform_remote_state" "infra" {
  backend = "local"
  config = {
    path = "../infra/terraform.tfstate"
  }
}

locals {
  ecr_repository_url     = data.terraform_remote_state.infra.outputs.ecr_repository_url
  apprunner_iam_role_arn = data.terraform_remote_state.infra.outputs.apprunner_iam_role_arn

  tags = {
    Project     = "seattle-energy-predictor"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# App Runner service serving the BentoML API
# Run this only after the Docker image has been pushed to ECR (terraform/infra)
resource "aws_apprunner_service" "seattle" {
  service_name = var.service_name

  source_configuration {
    image_repository {
      image_identifier      = "${local.ecr_repository_url}:${var.image_tag}"
      image_repository_type = "ECR"
      image_configuration {
        port = "3000"
      }
    }
    auto_deployments_enabled = false

    authentication_configuration {
      access_role_arn = local.apprunner_iam_role_arn
    }
  }

  instance_configuration {
    cpu    = "1 vCPU"
    memory = "2 GB"
  }

  tags = local.tags
}
