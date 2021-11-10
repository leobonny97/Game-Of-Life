#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include "mpi.h" //libreria di MPI

//funzione che ci permette di allocare la matrice dinamicamente e in celle contigue di memoria
//in questo modo possiamo passare le matrici con MPI senza avere problemi
bool **alloc_array_bool(int, int);

//funzione che ci permette di trovare il numero di vicini vivi --> ulteriore descrizione prima dell'implementazione
void trovaNumeroViciniVivi(bool **, int, int, int **, bool);

//funzione che ci permette di aggiornare lo stato delle cellule --> ulteriore descrizione prima dell'implementazione
void aggiornaStatoCellule(bool **, int, int, int **, bool);

//funzione che ci permette di stampare la matrice di cellule --> ulteriore descrizione prima dell'implementazione
void stampaMatrice(bool **, int, int);

//funzione che calcola il numero di elementi da inviare, il numero di righe e il displacement --> ulteriore descrizione prima dell'implementazione
void calcolaSendCountsAndDispls(int, int, int , int *, int*, int *);

int main(int argc, char *argv[]) {
    int my_rank; //identificativo del processo
    int p; //numero di processi
    MPI_Status status; //stato
    MPI_Request request1, request2; //request
    MPI_Init(&argc, &argv); //inizializziamo MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //permette di identificare il rank del processo
    MPI_Comm_size(MPI_COMM_WORLD, &p); //permette di definire il numero di processi in esecuzione

    if(argc == 4) { //se vengono passati tre argomenti
        int iterazioni = atoi(argv[1]); //recuperiamo il numero di iterazioni da effettuare
        int righe = atoi(argv[2]); //recuperiamo il numero di righe
        int colonne = atoi(argv[3]); //recuperiamo il numero di colonne
        bool **matrice = alloc_array_bool(righe, colonne); //allocazione dinamica della matrice di boolean
        if(my_rank == 0) {
            printf("Numero di iterazioni = %d.\n", iterazioni);
            printf("Numero di righe = %d.\n", righe);
            printf("Numero di colonne = %d.\n\n", colonne);
            srand(my_rank); //inizializzazione random generator           
            printf("Stampo la matrice generata.\n");
            for(int c = 0; c < righe; c++) {
                for(int i = 0; i < colonne; i++)
                {
                    matrice[c][i] = rand() % 2; //generiamo boolean random
                    printf("%d ", matrice[c][i]);
                }
                printf("\n");
            }
        }
        if(p == 1) { //esecuzione su un unico nodo
            float time = MPI_Wtime();
            printf("\nSto eseguendo su un unico nodo.\n");
            int **vicini_vivi = (int **)malloc(sizeof(int *)*righe); //allocazione dinamica della matrice contenente il numero di vicini vivi
            for(int i=0; i<righe; i++) {
                vicini_vivi[i] = (int *)malloc(sizeof(int)*colonne); //allocazione dinamica della matrice contenente il numero di vicini vivi
            }                 
            for(int c = 0; c < iterazioni; c++) {
                //printf("\nInizio iterazione %d.\n", c+1);
                trovaNumeroViciniVivi(matrice, righe, colonne, vicini_vivi, 1); //andiamo a ricercare il numero di vicini vivi
                aggiornaStatoCellule(matrice, righe, colonne, vicini_vivi, 1); //andiamo ad aggiornare lo stato delle cellule
                //stampaMatrice(matrice, righe, colonne); //stampiamo per verificare la correttezza
            }            
            float etime =  MPI_Wtime() - time;
            printf("\nStampo la matrice finale.\n");
            stampaMatrice(matrice, righe, colonne); //stampiamo la matrice finale
            printf("\nTime: %f\n\n", etime);
        } else { //esecuzione su più nodi
            float time;
            if(my_rank == 0) {
                printf("Sto eseguendo su più nodi.\n");
                time = MPI_Wtime();
            }            
            bool **matrice_rcv;
            int **vicini_vivi;
            if(righe < p) { //se il numero di righe è minore o uguale al numero di processi
                int ranks[righe], new_my_rank;
                int displs[righe]; //indica la posizione dove iniziare ad inviare
                int send_counts_row[righe]; //numero di righe da inviare e da ricevere
                int send_counts[righe]; //numero di elementi da inviare e da ricevere 
                MPI_Group orig_group, new_group; //nuovo gruppo
                MPI_Comm new_comm; //nuovo communicator
                MPI_Comm_group(MPI_COMM_WORLD, &orig_group); //estraiamo il gruppo originale
                for(int i = 0; i < righe; i++) {
                    ranks[i] = i;
                }
                MPI_Group_incl(orig_group, righe, ranks, &new_group); //creiamo il nuovo gruppo
                MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm); //creiamo il communicator per il nuovo gruppo
                MPI_Group_rank(new_group, &new_my_rank); //nuovo rank dei processi
                if (my_rank < righe) {                                       
                    calcolaSendCountsAndDispls(righe, colonne, righe, send_counts_row, send_counts, displs); //calcoliamo il numero di elementi da inviare e i displacements
                    matrice_rcv = alloc_array_bool(send_counts_row[new_my_rank]+2, colonne); //allocazione dinamica della matrice di boolean da ricevere
                    MPI_Scatterv(&(matrice[0][0]), send_counts, displs, MPI_C_BOOL, &(matrice_rcv[1][0]), send_counts[new_my_rank], MPI_C_BOOL, 0, new_comm); //distribuiamo la matrice iniziale
                    vicini_vivi = (int **)malloc(sizeof(int *)*send_counts_row[my_rank]); //allocazione dinamica della matrice contenente il numero di vicini vivi
                    for(int i=0; i<send_counts_row[new_my_rank]; i++) {
                        vicini_vivi[i] = (int *)malloc(sizeof(int)*colonne); //allocazione dinamica della matrice contenente il numero di vicini vivi
                    }   
                    for(int i = 0; i < iterazioni; i++) {
                        MPI_Isend(&(matrice_rcv[1][0]), colonne, MPI_C_BOOL, (new_my_rank+righe-1)%righe, 0, new_comm, &request1); //mandiamo la riga superiore aggiornata                                 
                        MPI_Isend(&(matrice_rcv[send_counts_row[new_my_rank]][0]), colonne, MPI_C_BOOL, (new_my_rank+1)%righe, 1, new_comm, &request2); //mandiamo la riga inferiore aggiornata  
                        MPI_Recv(&(matrice_rcv[send_counts_row[new_my_rank]+1][0]), colonne, MPI_C_BOOL, (new_my_rank+1)%righe, 0, new_comm, &status); //riceviamo la riga superiore aggiornata 
                        MPI_Recv(&(matrice_rcv[0][0]), colonne, MPI_C_BOOL, (new_my_rank+righe-1)%righe, 1, new_comm, &status); //riceviamo la riga inferiore aggiornata
                        trovaNumeroViciniVivi(matrice_rcv, send_counts_row[new_my_rank], colonne, vicini_vivi, 0); //andiamo a ricercare il numero di vicini vivi
                        aggiornaStatoCellule(matrice_rcv, send_counts_row[new_my_rank], colonne, vicini_vivi, 0); //andiamo ad aggiornare lo stato delle cellule
                    }
                    MPI_Gatherv(&(matrice_rcv[1][0]), send_counts[new_my_rank], MPI_C_BOOL, &(matrice[0][0]), send_counts, displs, MPI_C_BOOL, 0, new_comm); //otteniamo la matrice finale
                }              
            } else { //se il numero di righe è maggiore o uguale del numero di processi        
                int displs[p]; //indica la posizione dove iniziare ad inviare
                int send_counts_row[p]; //numero di righe da inviare e da ricevere
                int send_counts[p]; //numero di elementi da inviare e da ricevere       
                calcolaSendCountsAndDispls(righe, colonne, p, send_counts_row, send_counts, displs); //calcoliamo il numero di elementi da inviare e i displacements
                matrice_rcv = alloc_array_bool(send_counts_row[my_rank]+2, colonne); //allocazione dinamica della matrice di boolean da ricevere
                MPI_Scatterv(&(matrice[0][0]), send_counts, displs, MPI_C_BOOL, &(matrice_rcv[1][0]), send_counts[my_rank], MPI_C_BOOL, 0, MPI_COMM_WORLD); //distribuiamo la matrice iniziale
                vicini_vivi = (int **)malloc(sizeof(int *)*send_counts_row[my_rank]); //allocazione dinamica della matrice contenente il numero di vicini vivi
                for(int i=0; i<send_counts_row[my_rank]; i++) {
                    vicini_vivi[i] = (int *)malloc(sizeof(int)*colonne); //allocazione dinamica della matrice contenente il numero di vicini vivi
                }   
                for(int i = 0; i < iterazioni; i++) {
                    MPI_Isend(&(matrice_rcv[1][0]), colonne, MPI_C_BOOL, (my_rank+p-1)%p, 0, MPI_COMM_WORLD, &request1); //mandiamo la riga superiore aggiornata  
                    MPI_Isend(&(matrice_rcv[send_counts_row[my_rank]][0]), colonne, MPI_C_BOOL, (my_rank+1)%p, 1, MPI_COMM_WORLD, &request2); //mandiamo la riga inferiore aggiornata 
                    MPI_Recv(&(matrice_rcv[send_counts_row[my_rank]+1][0]), colonne, MPI_C_BOOL, (my_rank+1)%p, 0, MPI_COMM_WORLD, &status); //riceviamo la riga superiore aggiornata
                    MPI_Recv(&(matrice_rcv[0][0]), colonne, MPI_C_BOOL, (my_rank+p-1)%p, 1, MPI_COMM_WORLD, &status); //riceviamo la riga inferiore aggiornata
                    trovaNumeroViciniVivi(matrice_rcv, send_counts_row[my_rank], colonne, vicini_vivi, 0); //andiamo a ricercare il numero di vicini vivi
                    aggiornaStatoCellule(matrice_rcv, send_counts_row[my_rank], colonne, vicini_vivi, 0); //andiamo ad aggiornare lo stato delle cellule
                }
                MPI_Gatherv(&(matrice_rcv[1][0]), send_counts[my_rank], MPI_C_BOOL, &(matrice[0][0]), send_counts, displs, MPI_C_BOOL, 0, MPI_COMM_WORLD); //otteniamo la matrice finale
            }
            if(my_rank == 0) { //nodo root
                float etime =  MPI_Wtime() - time;
                stampaMatrice(matrice, righe, colonne);
                printf("\nTime: %f\n\n", etime);
            }
        }
    } else { //sono stati passati più argomenti o nessuno
        printf("Terminiamo perchè non è stato specificato il numero di iterazioni.\n");
    }  
    MPI_Finalize(); //termina MPI
    return 0;
}

//funzione che ci permette di trovare il numero di vicini vivi
//prende come argomento la matrice di cellule, il numero di righe della matrice, il numero di colonne della matrice e la matrice dove salveremo il numero di vicini vivi
//prende anche un boolean che indica se calcolare su tutte le righe (1) o se non contare la prima e l'ultima riga (0 in caso di esecuzione su più nodi)
//in particolare, flag sarà uguale a 1 se l'esecuzione sarà su un unico nodo, altrimenti sarà uguale a 0 
//non restituisce alcun valore e per ogni elemento della matrice di cellule, andiamo a vedere quanti tra gli 8 vicini sono vivi 
//infine aggiorniamo la matrice contentente il numero di vicini vivi per ogni elemento della matrice di cellule
//per verificare la correttezza ci facciamo stampare il numero sullo standard di output
void trovaNumeroViciniVivi(bool **matrice, int righe, int colonne, int **vicini_vivi, bool flag) {
    if(flag == 1) {
        for(int c = 0; c < righe; c++) {
            for(int i = 0; i < colonne; i++) {
                vicini_vivi[c][i] = 0;
                if(matrice[(c+1)%righe][i] == 1) { //in basso
                    vicini_vivi[c][i]++;
                }
                if(matrice[(c+righe-1)%righe][i] == 1) { //in alto
                    vicini_vivi[c][i]++;
                }
                if(matrice[c][(i+1)%colonne] == 1) { //a destra
                    vicini_vivi[c][i]++;
                }
                if(matrice[c][(i+colonne-1)%colonne] == 1) { //a sinistra
                    vicini_vivi[c][i]++;
                }
                if(matrice[(c+1)%righe][(i+1)%colonne] == 1) { //in basso a destra
                    vicini_vivi[c][i]++;
                }
                if(matrice[(c+righe-1)%righe][(i+1)%colonne] == 1) { //in alto a destra
                    vicini_vivi[c][i]++;            }
                if(matrice[(c+1)%righe][(i+colonne-1)%colonne] == 1) { //in basso a sinistra
                    vicini_vivi[c][i]++;
                }
                if(matrice[(c+righe-1)%righe][(i+colonne-1)%colonne] == 1) { //in alto a sinistra
                    vicini_vivi[c][i]++;
                }
                //printf("%d %d = Vicini vivi: %d.\n", c+1, i+1, vicini_vivi[c][i]);
            }
        }
    } else {
        for(int c = 0; c < righe; c++) {
            for(int i = 0; i < colonne; i++) {
                vicini_vivi[c][i] = 0;
                if(matrice[c+2][i] == 1) { //in basso
                    vicini_vivi[c][i]++;
                }
                if(matrice[c][i] == 1) { //in alto
                    vicini_vivi[c][i]++;
                }
                if(matrice[c+1][(i+1)%colonne] == 1) { //a destra
                    vicini_vivi[c][i]++;
                }
                if(matrice[c+1][(i+colonne-1)%colonne] == 1) { //a sinistra
                    vicini_vivi[c][i]++;
                }
                if(matrice[c+2][(i+1)%colonne] == 1) { //in basso a destra
                    vicini_vivi[c][i]++;
                }
                if(matrice[c][(i+1)%colonne] == 1) { //in alto a destra
                    vicini_vivi[c][i]++;            }
                if(matrice[c+2][(i+colonne-1)%colonne] == 1) { //in basso a sinistra
                    vicini_vivi[c][i]++;
                }
                if(matrice[c][(i+colonne-1)%colonne] == 1) { //in alto a sinistra
                    vicini_vivi[c][i]++;
                }
                //printf("%d %d = Vicini vivi: %d.\n", c+1, i+1, vicini_vivi[c][i]);
            }
        }
    }

    return;
}

//funzione che ci permette di aggiornare lo stato delle cellule
//prende come argomento la matrice di cellule, il numero di righe della matrice, il numero di colonne della matrice e la matrice che contiene il numero di vicini vivi
//prende anche un boolean che indica se calcolare su tutte le righe (1) o se non contare la prima e l'ultima riga (0 in caso di esecuzione su più nodi)
//in particolare, flag sarà uguale a 1 se l'esecuzione sarà su un unico nodo, altrimenti sarà uguale a 0 
//non restituisce alcun valore e per ogni elemento della matrice di cellule, andiamo a vedere il nuovo stato
//se una cellula è viva ed ha meno di due vicini vivi --> muore
//se una cellula è viva ed ha più di tre vicini vivi --> muore
//se una cellula è viva ed ha due o tre vivini vivi --> continua a vivere --> non abbiamo bisogno di implementare questo caso perchè in realtà non causa alcun aggiornamento
//se una cellula è morta ed ha esettamente tre vicini vivi --> torna a vivere
//per verificare la correttezza facciamo stampare 
void aggiornaStatoCellule(bool **matrice, int righe, int colonne, int **vicini_vivi, bool flag) {
    if(flag == 1) {
        for(int c = 0; c < righe; c++) {
            for(int i = 0; i < colonne; i++) {
                if(matrice[c][i] == 1 && vicini_vivi[c][i] < 2) { //cellula viva con meno di due vicini vivi
                    //printf("%d %d = Io sono %d - Vicini vivi: %d < 2 --> Muoio.\n", c+1, i+1, matrice[c][i], vicini_vivi[c][i]);
                    matrice[c][i] = 0; //muore
                } else if(matrice[c][i] == 1 && vicini_vivi[c][i] > 3) { //cellula viva con più di tre vicini vivi
                    //printf("%d %d = Io sono %d - Vicini vivi: %d > 3 --> Muoio.\n", c+1, i+1, matrice[c][i], vicini_vivi[c][i]);
                    matrice[c][i] = 0; //muore
                } else if(matrice[c][i] == 0 && vicini_vivi[c][i] == 3) { //cellula morta con esattamente tre vicini vivi
                    //printf("%d %d = Io sono %d - Vicini vivi: %d = 3 --> Rivivo.\n", c+1, i+1, matrice[c][i], vicini_vivi[c][i]);
                    matrice[c][i] = 1; //torna a vivere
                } else {
                    //printf("%d %d = Io sono %d - Vicini vivi: %d.\n", c+1, i+1, matrice[c][i], vicini_vivi[c][i]);
                }
                
            }
        }
    } else {
        for(int c = 0; c < righe; c++) {
            for(int i = 0; i < colonne; i++) {
                if(matrice[c+1][i] == 1 && vicini_vivi[c][i] < 2) { //cellula viva con meno di due vicini vivi
                    //printf("%d %d = Io sono %d - Vicini vivi: %d < 2 --> Muoio.\n", c+1, i+1, matrice[c][i], vicini_vivi[c][i]);
                    matrice[c+1][i] = 0; //muore
                } else if(matrice[c+1][i] == 1 && vicini_vivi[c][i] > 3) { //cellula viva con più di tre vicini vivi
                    //printf("%d %d = Io sono %d - Vicini vivi: %d > 3 --> Muoio.\n", c+1, i+1, matrice[c][i], vicini_vivi[c][i]);
                    matrice[c+1][i] = 0; //muore
                } else if(matrice[c+1][i] == 0 && vicini_vivi[c][i] == 3) { //cellula morta con esattamente tre vicini vivi
                    //printf("%d %d = Io sono %d - Vicini vivi: %d = 3 --> Rivivo.\n", c+1, i+1, matrice[c][i], vicini_vivi[c][i]);
                    matrice[c+1][i] = 1; //torna a vivere
                } else {
                    //printf("%d %d = Io sono %d - Vicini vivi: %d.\n", c+1, i+1, matrice[c][i], vicini_vivi[c][i]);
                }

            }
        }
    }
    return;
}

//funzione che ci permette di stampare la matrice di cellule
//prende come parametri la matrice di cellule da stampare, il numero di righe e il numero di colonne
void stampaMatrice(bool **matrice, int righe, int colonne) {
    printf("\n");
    for(int c = 0; c < righe; c++) {
        for(int i = 0; i < colonne; i++)
        {
            printf("%d ", matrice[c][i]);
        }
        printf("\n");
    }
}

//funzione che ci permette di allocare la matrice dinamicamente e in celle contigue di memoria
//in questo modo possiamo passare le matrici con MPI senza avere problemi
//prende come parametri il numero di righe e il numero di colonne
bool **alloc_array_bool(int righe, int colonne) {
    bool *data = (bool *)malloc(righe*colonne*sizeof(bool));
    bool **array= (bool **)malloc(righe*sizeof(bool*));
    for(int i=0; i<righe; i++)
        array[i] = &(data[colonne*i]);
    return array;
}

//funzione che calcola il numero di elementi da inviare, il numero di righe e il displacement
//prende come parametri il numero di righe della matrice originale, il numero di colonne della matrice originale e il numero di processi
//inoltre, prende le variabili dove salvare il numero di elementi da inviare, il numero di righe da inviare e i displacement per i vari processi 
void calcolaSendCountsAndDispls(int righe, int colonne, int p, int *send_counts_row, int *send_counts, int *displs) {
    int quoziente = righe / p;
    //printf("Il quoziente equivale a %d.\n", quoziente);
    int resto = righe % p;
    //printf("Il resto equivale a %d.\n", resto);
    for(int i = 0; i < p; i++) {
        if(resto > 0) {
            send_counts_row[i] = quoziente + 1;           
            resto--;
        } else {
            send_counts_row[i] = quoziente;
        }
        send_counts[i] = send_counts_row[i] * colonne;
        if(i != 0) {
            displs[i] = displs[i-1] + (send_counts[i-1]);
        } else {
            displs[0] = 0;
        }
    }
}