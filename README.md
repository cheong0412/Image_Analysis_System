# AI 기반 이미지 분석: 범인은 바로 너!
## AI 기반 얼굴 이미지 분석을 통한 용의자 식별 시스템

## 목차
  - [프로젝트 기본 정보](#프로젝트-기본-정보)
  - [프로젝트 개요](#프로젝트-개요)
  - [프로젝트 설명](#프로젝트-설명)
  - [분석 결과](#분석-결과)
  - [기대 효과](#기대-효과)

## 프로젝트 기본 정보
- 프로젝트 이름: AI 기반 이미지 분석: 범인은 바로 너!
- 프로젝트 참여인원: 명
- 사용 언어: Python
- 담당 역할
  - 마스크 착용/미착용 이미지 수집 및 전처리
  - 이미지 데이터 전처리 수행
  - CNN 기반 마스크 착용 여부 이진 분류 모델 개발
  - 발표 자료 제작 및 프로젝트 발표

## 프로젝트 개요
실제 사건을 가상 시나리오로 구성하여 각 단서를 분리된 모델로 분석하고 통합 추론을 통해 ‘용의자’를 특정하는 과정을 구현함.
딥러닝 모델(CNN 기반)을 직접 설계·학습하고, 데이터 수집부터 전처리, 시각화, 결과 통합까지 전 과정을 수행함.

## 프로젝트 설명
가상의 범인(얼굴 이미지)을 설정하 4개의 단서(마스츠 착용 여부, 성별, 표정, 손 글씨)를 통해 AI 기반 얼굴 이미지 분석을 통한 용의자 식별 시스템

## 분석 결과
<img width="312" height="126" alt="1-1 마스크모델" src="https://github.com/user-attachments/assets/f1377786-4ab1-4bfc-89f9-153e4a052659" />
<img width="822" height="331" alt="1-2 마스크 모델" src="https://github.com/user-attachments/assets/3fcbc316-0548-45fa-9520-6aff43ddae1e" />
<img width="717" height="437" alt="4 마스크모델 결과" src="https://github.com/user-attachments/assets/0265e380-a9d5-4124-96f2-73e41dc82406" />
<img width="990" height="461" alt="5  성별 모델" src="https://github.com/user-attachments/assets/1f17f2a6-4daf-41c5-a310-686944879798" />
<img width="862" height="567" alt="6 성별모델 결과" src="https://github.com/user-attachments/assets/a833cb37-40df-4c5c-9e03-bddb0fc80962" />
<img width="896" height="437" alt="8 감정모델" src="https://github.com/user-attachments/assets/d7f5f997-5813-40f2-a85e-5d906a5fee04" />
<img width="688" height="408" alt="9 감정모델예측" src="https://github.com/user-attachments/assets/b586dc07-66f1-491e-9e9d-f950aa8dce1c" />

 <img width="1023" height="402" alt="4-1  글씨모델" src="https://github.com/user-attachments/assets/d5a12eb3-4734-4d77-8de2-fa1ac75fc1fa" />
<img width="700" height="413" alt="4-2 글씨모델 결과" src="https://github.com/user-attachments/assets/e542d1ec-f488-41fe-a550-bdb0c4573511" />

## 기대 효과
- 마스크 착용 여부, 성별, 감정, 손글씨 식별 모델은 다양한 웹/앱 서비스에 적용 가능함.
- 코로나19와 같은 감염병 유행 시기에는 마스크 착용 감지 기능이 방역 관리에 활용될 수 있음.
- 성별 분류 모델은 공항 입국심사 등 신원 확인이 필요한 보안 시스템에 적용될 수 있음.
- 감정 분석 모델은 고객 응대 서비스에서 사용자 감정 파악에 활용될 수 있음.
- 손글씨 분석은 디지털 필기 분석 등에 적용 가능

## Lesson & Learned
- 딥러닝 모델 개발에 대한 이해는 부족했으나, 프로젝트를 통해 전체적인 흐름과 주요 용어에 익숙해짐.
- 이미지 데이터 분석에 어려움이 있었으나, 실습을 통해 이미지 처리 방식과 모델 학습 과정을 이해하게 됨.
- CNN 모델 구조와 계층별 역할에 대해 기초적인 이해를 쌓는 계기가 됨.
