--
-- PostgreSQL database dump
--

\restrict jpeor1JXxZv4Efix6evcl0cTUaIg9NsqeVVBjOQM0BLZujegjm0jabwKNMiv1Pv

-- Dumped from database version 18.1
-- Dumped by pg_dump version 18.1

-- Started on 2026-02-13 01:11:53

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 226 (class 1259 OID 16447)
-- Name: bookings; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.bookings (
    id integer NOT NULL,
    user_id integer NOT NULL,
    hostel_id integer NOT NULL,
    check_in date NOT NULL,
    check_out date NOT NULL,
    status character varying(20) DEFAULT 'pending'::character varying,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.bookings OWNER TO postgres;

--
-- TOC entry 225 (class 1259 OID 16446)
-- Name: bookings_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.bookings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.bookings_id_seq OWNER TO postgres;

--
-- TOC entry 4988 (class 0 OID 0)
-- Dependencies: 225
-- Name: bookings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.bookings_id_seq OWNED BY public.bookings.id;


--
-- TOC entry 230 (class 1259 OID 16498)
-- Name: complaints; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.complaints (
    id integer NOT NULL,
    user_id integer NOT NULL,
    hostel_id integer NOT NULL,
    category character varying(50) NOT NULL,
    severity character varying(10) NOT NULL,
    status character varying(20) DEFAULT 'open'::character varying NOT NULL,
    assigned_to integer,
    description text NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    resolved_at timestamp without time zone,
    CONSTRAINT complaints_severity_check CHECK (((severity)::text = ANY ((ARRAY['low'::character varying, 'medium'::character varying, 'high'::character varying])::text[]))),
    CONSTRAINT complaints_status_check CHECK (((status)::text = ANY ((ARRAY['open'::character varying, 'in_progress'::character varying, 'resolved'::character varying, 'closed'::character varying])::text[])))
);


ALTER TABLE public.complaints OWNER TO postgres;

--
-- TOC entry 229 (class 1259 OID 16497)
-- Name: complaints_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.complaints_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.complaints_id_seq OWNER TO postgres;

--
-- TOC entry 4989 (class 0 OID 0)
-- Dependencies: 229
-- Name: complaints_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.complaints_id_seq OWNED BY public.complaints.id;


--
-- TOC entry 222 (class 1259 OID 16408)
-- Name: hostels; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hostels (
    id integer NOT NULL,
    name character varying(150) NOT NULL,
    address text NOT NULL,
    city character varying(100) NOT NULL,
    total_capacity integer NOT NULL,
    available_capacity integer NOT NULL,
    description text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.hostels OWNER TO postgres;

--
-- TOC entry 221 (class 1259 OID 16407)
-- Name: hostels_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.hostels_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hostels_id_seq OWNER TO postgres;

--
-- TOC entry 4990 (class 0 OID 0)
-- Dependencies: 221
-- Name: hostels_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.hostels_id_seq OWNED BY public.hostels.id;


--
-- TOC entry 224 (class 1259 OID 16424)
-- Name: interactions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.interactions (
    id integer NOT NULL,
    user_id integer NOT NULL,
    hostel_id integer,
    action_type character varying(50) NOT NULL,
    metadata jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.interactions OWNER TO postgres;

--
-- TOC entry 223 (class 1259 OID 16423)
-- Name: interactions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.interactions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.interactions_id_seq OWNER TO postgres;

--
-- TOC entry 4991 (class 0 OID 0)
-- Dependencies: 223
-- Name: interactions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.interactions_id_seq OWNED BY public.interactions.id;


--
-- TOC entry 228 (class 1259 OID 16471)
-- Name: reviews; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.reviews (
    id integer NOT NULL,
    user_id integer NOT NULL,
    hostel_id integer NOT NULL,
    overall_rating integer,
    cleanliness integer,
    facilities integer,
    management integer,
    text_review text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT reviews_cleanliness_check CHECK (((cleanliness >= 1) AND (cleanliness <= 5))),
    CONSTRAINT reviews_facilities_check CHECK (((facilities >= 1) AND (facilities <= 5))),
    CONSTRAINT reviews_management_check CHECK (((management >= 1) AND (management <= 5))),
    CONSTRAINT reviews_overall_rating_check CHECK (((overall_rating >= 1) AND (overall_rating <= 5)))
);


ALTER TABLE public.reviews OWNER TO postgres;

--
-- TOC entry 227 (class 1259 OID 16470)
-- Name: reviews_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.reviews_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.reviews_id_seq OWNER TO postgres;

--
-- TOC entry 4992 (class 0 OID 0)
-- Dependencies: 227
-- Name: reviews_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.reviews_id_seq OWNED BY public.reviews.id;


--
-- TOC entry 220 (class 1259 OID 16389)
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id integer NOT NULL,
    name character varying(100) NOT NULL,
    email character varying(150) NOT NULL,
    password_hash text NOT NULL,
    role character varying(20) NOT NULL,
    phone character varying(20),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT users_role_check CHECK (((role)::text = ANY ((ARRAY['student'::character varying, 'warden'::character varying, 'admin'::character varying])::text[])))
);


ALTER TABLE public.users OWNER TO postgres;

--
-- TOC entry 219 (class 1259 OID 16388)
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO postgres;

--
-- TOC entry 4993 (class 0 OID 0)
-- Dependencies: 219
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- TOC entry 4786 (class 2604 OID 16450)
-- Name: bookings id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bookings ALTER COLUMN id SET DEFAULT nextval('public.bookings_id_seq'::regclass);


--
-- TOC entry 4791 (class 2604 OID 16501)
-- Name: complaints id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.complaints ALTER COLUMN id SET DEFAULT nextval('public.complaints_id_seq'::regclass);


--
-- TOC entry 4782 (class 2604 OID 16411)
-- Name: hostels id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hostels ALTER COLUMN id SET DEFAULT nextval('public.hostels_id_seq'::regclass);


--
-- TOC entry 4784 (class 2604 OID 16427)
-- Name: interactions id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.interactions ALTER COLUMN id SET DEFAULT nextval('public.interactions_id_seq'::regclass);


--
-- TOC entry 4789 (class 2604 OID 16474)
-- Name: reviews id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews ALTER COLUMN id SET DEFAULT nextval('public.reviews_id_seq'::regclass);


--
-- TOC entry 4780 (class 2604 OID 16392)
-- Name: users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- TOC entry 4978 (class 0 OID 16447)
-- Dependencies: 226
-- Data for Name: bookings; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.bookings (id, user_id, hostel_id, check_in, check_out, status, created_at) FROM stdin;
1	1	1	2026-03-01	2026-06-30	confirmed	2026-02-12 23:46:23.621695
\.


--
-- TOC entry 4982 (class 0 OID 16498)
-- Dependencies: 230
-- Data for Name: complaints; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.complaints (id, user_id, hostel_id, category, severity, status, assigned_to, description, created_at, resolved_at) FROM stdin;
1	1	1	electricity	medium	open	\N	Electricity issue interrupted last night	2026-02-13 00:42:31.187557	\N
2	1	1	electricity	medium	open	2	Electricity issue interrupted last night	2026-02-13 00:50:50.780631	\N
\.


--
-- TOC entry 4974 (class 0 OID 16408)
-- Dependencies: 222
-- Data for Name: hostels; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hostels (id, name, address, city, total_capacity, available_capacity, description, created_at) FROM stdin;
1	Islamia Girls Hostel	Near Fast University, Islamabad	Islamabad	180	40	Safe and secure girls hostel with mess and WiFi	2026-02-12 22:53:13.869839
2	Fatima Girls Hostel	G13/4 street 130 house number 26	Islamabad	120	25	Affordable hostel with laundry and study rooms	2026-02-12 22:53:13.869839
\.


--
-- TOC entry 4976 (class 0 OID 16424)
-- Dependencies: 224
-- Data for Name: interactions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.interactions (id, user_id, hostel_id, action_type, metadata, created_at) FROM stdin;
1	1	1	view	{"source": "homepage"}	2026-02-12 22:56:56.376215
2	1	2	favorite	{"source": "search_result"}	2026-02-12 22:56:56.376215
\.


--
-- TOC entry 4980 (class 0 OID 16471)
-- Dependencies: 228
-- Data for Name: reviews; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.reviews (id, user_id, hostel_id, overall_rating, cleanliness, facilities, management, text_review, created_at) FROM stdin;
1	1	1	5	5	4	5	Very clean and secure hostel. Staff is cooperative.	2026-02-13 00:13:47.344424
3	1	1	5	5	4	5	Very clean and secure hostel. Staff is cooperative.	2026-02-13 00:18:17.927348
\.


--
-- TOC entry 4972 (class 0 OID 16389)
-- Dependencies: 220
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (id, name, email, password_hash, role, phone, created_at) FROM stdin;
1	Zarnab Student	zarnab@student.com	dummyhash1	student	03001234567	2026-02-12 22:51:17.571766
2	Warden Fatima	fatima@warden.com	dummyhash2	warden	03007654321	2026-02-12 22:51:17.571766
\.


--
-- TOC entry 4994 (class 0 OID 0)
-- Dependencies: 225
-- Name: bookings_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.bookings_id_seq', 1, true);


--
-- TOC entry 4995 (class 0 OID 0)
-- Dependencies: 229
-- Name: complaints_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.complaints_id_seq', 2, true);


--
-- TOC entry 4996 (class 0 OID 0)
-- Dependencies: 221
-- Name: hostels_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.hostels_id_seq', 2, true);


--
-- TOC entry 4997 (class 0 OID 0)
-- Dependencies: 223
-- Name: interactions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.interactions_id_seq', 2, true);


--
-- TOC entry 4998 (class 0 OID 0)
-- Dependencies: 227
-- Name: reviews_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.reviews_id_seq', 3, true);


--
-- TOC entry 4999 (class 0 OID 0)
-- Dependencies: 219
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.users_id_seq', 2, true);


--
-- TOC entry 4810 (class 2606 OID 16459)
-- Name: bookings bookings_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bookings
    ADD CONSTRAINT bookings_pkey PRIMARY KEY (id);


--
-- TOC entry 4814 (class 2606 OID 16516)
-- Name: complaints complaints_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.complaints
    ADD CONSTRAINT complaints_pkey PRIMARY KEY (id);


--
-- TOC entry 4806 (class 2606 OID 16422)
-- Name: hostels hostels_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hostels
    ADD CONSTRAINT hostels_pkey PRIMARY KEY (id);


--
-- TOC entry 4808 (class 2606 OID 16435)
-- Name: interactions interactions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.interactions
    ADD CONSTRAINT interactions_pkey PRIMARY KEY (id);


--
-- TOC entry 4812 (class 2606 OID 16486)
-- Name: reviews reviews_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews
    ADD CONSTRAINT reviews_pkey PRIMARY KEY (id);


--
-- TOC entry 4802 (class 2606 OID 16405)
-- Name: users users_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_key UNIQUE (email);


--
-- TOC entry 4804 (class 2606 OID 16403)
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- TOC entry 4817 (class 2606 OID 16465)
-- Name: bookings fk_booking_hostel; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bookings
    ADD CONSTRAINT fk_booking_hostel FOREIGN KEY (hostel_id) REFERENCES public.hostels(id) ON DELETE CASCADE;


--
-- TOC entry 4818 (class 2606 OID 16460)
-- Name: bookings fk_booking_user; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bookings
    ADD CONSTRAINT fk_booking_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- TOC entry 4821 (class 2606 OID 16527)
-- Name: complaints fk_complaint_assigned; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.complaints
    ADD CONSTRAINT fk_complaint_assigned FOREIGN KEY (assigned_to) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- TOC entry 4822 (class 2606 OID 16522)
-- Name: complaints fk_complaint_hostel; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.complaints
    ADD CONSTRAINT fk_complaint_hostel FOREIGN KEY (hostel_id) REFERENCES public.hostels(id) ON DELETE CASCADE;


--
-- TOC entry 4823 (class 2606 OID 16517)
-- Name: complaints fk_complaint_user; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.complaints
    ADD CONSTRAINT fk_complaint_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- TOC entry 4815 (class 2606 OID 16441)
-- Name: interactions fk_hostel; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.interactions
    ADD CONSTRAINT fk_hostel FOREIGN KEY (hostel_id) REFERENCES public.hostels(id) ON DELETE CASCADE;


--
-- TOC entry 4819 (class 2606 OID 16492)
-- Name: reviews fk_review_hostel; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews
    ADD CONSTRAINT fk_review_hostel FOREIGN KEY (hostel_id) REFERENCES public.hostels(id) ON DELETE CASCADE;


--
-- TOC entry 4820 (class 2606 OID 16487)
-- Name: reviews fk_review_user; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews
    ADD CONSTRAINT fk_review_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- TOC entry 4816 (class 2606 OID 16436)
-- Name: interactions fk_user; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.interactions
    ADD CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


-- Completed on 2026-02-13 01:12:01

--
-- PostgreSQL database dump complete
--

\unrestrict jpeor1JXxZv4Efix6evcl0cTUaIg9NsqeVVBjOQM0BLZujegjm0jabwKNMiv1Pv

