import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { VitalSign } from './model/vital-sign';
import {Observable} from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  apiURL: string;

  constructor(private httpClient: HttpClient) {
    this.apiURL = 'http://localhost:5000';
  }


  public getPrediction(vitalSigns: VitalSign) {
   // console.log("`${this.apiURL}/predict/`", `${this.apiURL}/predict/`);
    console.log(vitalSigns)
    return this.httpClient.post(`${this.apiURL}/predict`, vitalSigns);
  }

  public getICDList(){
    return this.httpClient.get("./assets/icd10Diagnosis.json");
  }
}
