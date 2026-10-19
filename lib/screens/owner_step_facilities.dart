import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_steps_mess_safety_rules.dart';

class OwnerStepFacilities extends StatefulWidget {
  final String ownerEmail;
  final String ownerName;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;

  final double latitude;
  final double longitude;

  final int totalFloors;
  final int totalRooms;
  final int totalCapacity;
  final List<Map<String, dynamic>> roomTypes;

  const OwnerStepFacilities({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
    required this.latitude,
    required this.longitude,
    required this.totalFloors,
    required this.totalRooms,
    required this.totalCapacity,
    required this.roomTypes,
  });

  @override
  State<OwnerStepFacilities> createState() =>
      _OwnerStepFacilitiesState();
}

class _OwnerStepFacilitiesState
    extends State<OwnerStepFacilities>
    with SingleTickerProviderStateMixin {
  // ===========================================================================
  // FACILITIES
  // ===========================================================================

  final Map<String, List<Map<String, dynamic>>> _cats = {
    'Internet & Tech': [
      {
        'label': 'WiFi',
        'icon': Icons.wifi_rounded,
        'sel': true,
      },
      {
        'label': 'Cable TV',
        'icon': Icons.tv_rounded,
        'sel': false,
      },
    ],

    'Comfort': [
      {
        'label': 'AC',
        'icon': Icons.ac_unit_rounded,
        'sel': false,
      },
      {
        'label': 'Heater',
        'icon': Icons.thermostat_rounded,
        'sel': false,
      },
      {
        'label': 'Attached Bath',
        'icon': Icons.bathroom_rounded,
        'sel': false,
      },
      {
        'label': 'Cupboard',
        'icon': Icons.inventory_2_rounded,
        'sel': false,
      },
    ],

    'Utilities': [
      {
        'label': 'Hot Water',
        'icon': Icons.water_drop_rounded,
        'sel': true,
      },
      {
        'label': 'Generator',
        'icon': Icons.bolt_rounded,
        'sel': false,
      },
      {
        'label': 'Water Filter',
        'icon': Icons.water_rounded,
        'sel': false,
      },
      {
        'label': 'Laundry',
        'icon':
            Icons.local_laundry_service_rounded,
        'sel': false,
      },
    ],

    'Common Areas': [
      {
        'label': 'Study Room',
        'icon': Icons.menu_book_rounded,
        'sel': false,
      },
      {
        'label': 'Common Room',
        'icon': Icons.weekend_rounded,
        'sel': false,
      },
      {
        'label': 'Cafeteria',
        'icon': Icons.restaurant_rounded,
        'sel': false,
      },
      {
        'label': 'Library',
        'icon': Icons.local_library_rounded,
        'sel': false,
      },
    ],

    'Safety': [
      {
        'label': 'CCTV',
        'icon': Icons.videocam_rounded,
        'sel': false,
      },
      {
        'label': 'Security Guard',
        'icon': Icons.security_rounded,
        'sel': false,
      },
      {
        'label': 'Prayer Room',
        'icon': Icons.mosque_rounded,
        'sel': false,
      },
    ],

    'Transport': [
      {
        'label': 'Parking',
        'icon':
            Icons.local_parking_rounded,
        'sel': false,
      },
      {
        'label': 'Bus Stop Nearby',
        'icon':
            Icons.directions_bus_rounded,
        'sel': false,
      },
    ],
  };

  final _customCtrl =
      TextEditingController();

  final _customFocus =
      FocusNode();

  final List<String> _customList = [];

  String? _error;

  late final AnimationController _anim;

  late final Animation<double> _fade;

  late final Animation<double> _slide;

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  @override
  void initState() {
    super.initState();

    _anim = AnimationController(
      vsync: this,
      duration:
          const Duration(
        milliseconds: 750,
      ),
    );

    _fade = CurvedAnimation(
      parent: _anim,
      curve: Curves.easeOut,
    );

    _slide = Tween<double>(
      begin: 30,
      end: 0,
    ).animate(
      CurvedAnimation(
        parent: _anim,
        curve:
            Curves.easeOutCubic,
      ),
    );

    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();

    _customCtrl.dispose();
    _customFocus.dispose();

    super.dispose();
  }

  // ===========================================================================
  // SELECTED COUNT
  // ===========================================================================

  int get _selectedCount {
    final selected =
        _cats.values
            .expand(
              (items) => items,
            )
            .where(
              (item) =>
                  item['sel'] == true,
            )
            .length;

    return selected +
        _customList.length;
  }

  // ===========================================================================
  // CATEGORY COLORS
  // ===========================================================================

  Color _catColor(
    String category,
  ) {
    switch (category) {
      case 'Internet & Tech':
        return AppTheme.tealMint;

      case 'Comfort':
        return const Color(
          0xFFC49AFF,
        );

      case 'Utilities':
        return const Color(
          0xFFFFB76A,
        );

      case 'Common Areas':
        return const Color(
          0xFF78B7FF,
        );

      case 'Safety':
        return const Color(
          0xFFFF8585,
        );

      case 'Transport':
        return const Color(
          0xFF9DA8FF,
        );

      default:
        return AppTheme.tealMint;
    }
  }

  IconData _categoryIcon(
    String category,
  ) {
    switch (category) {
      case 'Internet & Tech':
        return Icons.wifi_rounded;

      case 'Comfort':
        return Icons.chair_alt_rounded;

      case 'Utilities':
        return Icons.bolt_rounded;

      case 'Common Areas':
        return Icons.groups_rounded;

      case 'Safety':
        return Icons.shield_outlined;

      case 'Transport':
        return Icons.directions_bus_rounded;

      default:
        return Icons.star_outline_rounded;
    }
  }

  // ===========================================================================
  // CUSTOM FACILITY
  // ===========================================================================

  void _addCustomFacility() {
    final value =
        _customCtrl.text.trim();

    if (value.isEmpty) {
      setState(() {
        _error =
            'Enter a facility name first.';
      });

      return;
    }

    final alreadyExists =
        _customList.any(
          (item) =>
              item.toLowerCase() ==
              value.toLowerCase(),
        );

    final standardExists =
        _cats.values
            .expand(
              (items) => items,
            )
            .any(
              (item) =>
                  item['label']
                      .toString()
                      .toLowerCase() ==
                  value.toLowerCase(),
            );

    if (alreadyExists ||
        standardExists) {
      setState(() {
        _error =
            'This facility is already listed.';
      });

      return;
    }

    setState(() {
      _customList.add(value);

      _customCtrl.clear();

      _error = null;
    });

    FocusScope.of(context)
        .unfocus();
  }

  // ===========================================================================
  // NEXT
  // ===========================================================================

  void _next() {
    FocusScope.of(context)
        .unfocus();

    if (_selectedCount == 0) {
      setState(() {
        _error =
            'Please select at least one facility.';
      });

      return;
    }

    final selectedFacilities = _cats.values
        .expand((items) => items)
        .where((item) => item['sel'] == true)
        .map((item) => item['label'].toString())
        .followedBy(_customList)
        .toList();

    setState(() {
      _error = null;
    });

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) =>
            OwnerStepMess(
          ownerEmail:
              widget.ownerEmail,
          ownerName:
              widget.ownerName,
          hostelName:
              widget.hostelName,
          hostelType:
              widget.hostelType,
          city:
              widget.city,
          address:
              widget.address,
          latitude:
              widget.latitude,
          longitude:
              widget.longitude,
          totalFloors:
              widget.totalFloors,
          totalRooms:
              widget.totalRooms,
          totalCapacity:
              widget.totalCapacity,
          roomTypes:
              widget.roomTypes,
          facilities:
              selectedFacilities,
        ),
      ),
    );
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(
    BuildContext context,
  ) {
    final keyboardHeight =
        MediaQuery.viewInsetsOf(
      context,
    ).bottom;

    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset:
            true,
        backgroundColor:
            Colors.transparent,
        body: VideoBackground(
          assetPath:
              'assets/images/background.mp4',
          overlayOpacity: 0.28,
          cardMargin:
              const EdgeInsets.all(
            14,
          ),
          cardRadius: 32,
          child: SafeArea(
            child:
                SingleChildScrollView(
              physics:
                  const BouncingScrollPhysics(),
              keyboardDismissBehavior:
                  ScrollViewKeyboardDismissBehavior
                      .onDrag,
              padding:
                  EdgeInsets.fromLTRB(
                18,
                12,
                18,
                22 +
                    keyboardHeight,
              ),
              child:
                  AnimatedBuilder(
                animation: _anim,
                builder: (
                  context,
                  child,
                ) {
                  return Transform
                      .translate(
                    offset: Offset(
                      0,
                      _slide.value,
                    ),
                    child: Opacity(
                      opacity:
                          _fade.value,
                      child: child,
                    ),
                  );
                },
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment
                          .start,
                  children: [
                    _buildHeader(),

                    const SizedBox(
                      height: 18,
                    ),

                    _buildMainCard(),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // HEADER
  // ===========================================================================

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            InkWell(
              onTap: () =>
                  Navigator.maybePop(
                context,
              ),
              borderRadius:
                  BorderRadius.circular(
                50,
              ),
              child: Container(
                width: 42,
                height: 42,
                decoration:
                    BoxDecoration(
                  color:
                      const Color(
                    0xFF02383B,
                  ).withOpacity(
                    0.55,
                  ),
                  shape:
                      BoxShape.circle,
                  border:
                      Border.all(
                    color: Colors
                        .white
                        .withOpacity(
                      0.22,
                    ),
                  ),
                ),
                child:
                    const Icon(
                  Icons
                      .arrow_back_rounded,
                  color:
                      Colors.white,
                  size: 20,
                ),
              ),
            ),

            const Spacer(),

            Container(
              padding:
                  const EdgeInsets
                      .symmetric(
                horizontal: 10,
                vertical: 6,
              ),
              decoration:
                  BoxDecoration(
                color:
                    const Color(
                  0xFF02383B,
                ).withOpacity(
                  0.58,
                ),
                borderRadius:
                    BorderRadius
                        .circular(
                  17,
                ),
                border:
                    Border.all(
                  color: AppTheme
                      .tealMint
                      .withOpacity(
                    0.24,
                  ),
                ),
              ),
              child: Row(
                mainAxisSize:
                    MainAxisSize.min,
                children: [
                  Text(
                    'Step 6 of 9',
                    style:
                        GoogleFonts.dmSans(
                      color:
                          Colors.white,
                      fontSize:
                          9,
                      fontWeight:
                          FontWeight
                              .w700,
                    ),
                  ),

                  const SizedBox(
                    width: 6,
                  ),

                  Container(
                    width: 4,
                    height: 4,
                    decoration:
                        BoxDecoration(
                      color: Colors
                          .white
                          .withOpacity(
                        0.45,
                      ),
                      shape:
                          BoxShape.circle,
                    ),
                  ),

                  const SizedBox(
                    width: 6,
                  ),

                  Text(
                    '$_selectedCount selected',
                    style:
                        GoogleFonts.dmSans(
                      color: AppTheme
                          .tealMint,
                      fontSize:
                          9,
                      fontWeight:
                          FontWeight
                              .w700,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),

        const SizedBox(
          height: 16,
        ),

        _buildProgress(),

        const SizedBox(
          height: 16,
        ),

        Text(
          'Facilities &\nAmenities',
          style:
              GoogleFonts
                  .playfairDisplay(
            fontSize: 35,
            height: 1.03,
            fontWeight:
                FontWeight.w700,
            color:
                Colors.white,
          ),
        ),

        const SizedBox(
          height: 12,
        ),

        Text(
          'Choose the facilities available at your hostel.',
          style:
              GoogleFonts.dmSans(
            color: Colors.white
                .withOpacity(
              0.70,
            ),
            fontSize: 12.5,
            height: 1.4,
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // PROGRESS
  // ===========================================================================

  Widget _buildProgress() {
    return Row(
      children:
          List.generate(
        9,
        (index) {
          final active =
              index < 6;

          return Expanded(
            child: Container(
              height:
                  active ? 4 : 3,
              margin:
                  EdgeInsets.only(
                right:
                    index < 8
                        ? 5
                        : 0,
              ),
              decoration:
                  BoxDecoration(
                color:
                    active
                        ? AppTheme
                            .tealMint
                        : Colors
                            .white
                            .withOpacity(
                            0.22,
                          ),
                borderRadius:
                    BorderRadius
                        .circular(
                  3,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  // ===========================================================================
  // MAIN CARD
  // ===========================================================================

  Widget _buildMainCard() {
    return Container(
      width: double.infinity,
      padding:
          const EdgeInsets.fromLTRB(
        18,
        16,
        18,
        20,
      ),
      decoration:
          BoxDecoration(
        gradient:
            LinearGradient(
          begin:
              Alignment.topLeft,
          end:
              Alignment.bottomRight,
          colors: [
            const Color(
              0xFF063F42,
            ).withOpacity(
              0.90,
            ),

            const Color(
              0xFF023336,
            ).withOpacity(
              0.90,
            ),

            const Color(
              0xFF021F22,
            ).withOpacity(
              0.92,
            ),
          ],
        ),
        borderRadius:
            BorderRadius.circular(
          28,
        ),
        border:
            Border.all(
          color: AppTheme
              .tealMint
              .withOpacity(
            0.22,
          ),
        ),
        boxShadow: [
          BoxShadow(
            color:
                Colors.black
                    .withOpacity(
              0.28,
            ),
            blurRadius: 26,
            offset:
                const Offset(
              0,
              11,
            ),
          ),
        ],
      ),
      child:
          Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          // ===============================================================
          // HANDLE
          // ===============================================================

          Center(
            child: Container(
              width: 38,
              height: 4,
              margin:
                  const EdgeInsets.only(
                bottom: 16,
              ),
              decoration:
                  BoxDecoration(
                color: AppTheme
                    .tealMint
                    .withOpacity(
                  0.30,
                ),
                borderRadius:
                    BorderRadius
                        .circular(
                  3,
                ),
              ),
            ),
          ),

          // ===============================================================
          // CARD TITLE
          // ===============================================================

          _buildCardHeader(),

          const SizedBox(
            height: 14,
          ),

          // ===============================================================
          // HOSTEL SUMMARY
          // ===============================================================

          _buildHostelSummary(),

          const SizedBox(
            height: 18,
          ),

          // ===============================================================
          // FACILITY CATEGORIES
          // ===============================================================

          ..._cats.entries.map(
            (entry) {
              return Padding(
                padding:
                    const EdgeInsets.only(
                  bottom: 18,
                ),
                child:
                    _buildCategory(
                  entry.key,
                  entry.value,
                ),
              );
            },
          ),

          // ===============================================================
          // CUSTOM FACILITIES
          // ===============================================================

          _buildCustomSection(),

          if (_error != null) ...[
            const SizedBox(
              height: 13,
            ),
            _buildError(),
          ],

          const SizedBox(
            height: 18,
          ),

          // ===============================================================
          // NEXT
          // ===============================================================

          _buildNextButton(),
        ],
      ),
    );
  }

  // ===========================================================================
  // CARD HEADER
  // ===========================================================================

  Widget _buildCardHeader() {
    return Row(
      children: [
        Container(
          width: 44,
          height: 44,
          decoration:
              BoxDecoration(
            color:
                AppTheme.teal
                    .withOpacity(
              0.22,
            ),
            borderRadius:
                BorderRadius.circular(
              13,
            ),
            border:
                Border.all(
              color: AppTheme
                  .tealMint
                  .withOpacity(
                0.18,
              ),
            ),
          ),
          child:
              const Icon(
            Icons
                .hotel_class_rounded,
            color:
                AppTheme.tealMint,
            size: 21,
          ),
        ),

        const SizedBox(
          width: 11,
        ),

        Expanded(
          child:
              Column(
            crossAxisAlignment:
                CrossAxisAlignment
                    .start,
            children: [
              Text(
                'Hostel Facilities',
                maxLines: 1,
                overflow:
                    TextOverflow
                        .ellipsis,
                style:
                    GoogleFonts
                        .playfairDisplay(
                  color:
                      Colors.white,
                  fontSize: 19,
                  fontWeight:
                      FontWeight
                          .w700,
                ),
              ),

              const SizedBox(
                height: 2,
              ),

              Text(
                'Select everything your hostel provides',
                maxLines: 1,
                overflow:
                    TextOverflow
                        .ellipsis,
                style:
                    GoogleFonts.dmSans(
                  color: Colors
                      .white
                      .withOpacity(
                    0.50,
                  ),
                  fontSize: 8.8,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(
          width: 8,
        ),

        Container(
          padding:
              const EdgeInsets
                  .symmetric(
            horizontal: 9,
            vertical: 6,
          ),
          decoration:
              BoxDecoration(
            color:
                AppTheme.teal
                    .withOpacity(
              0.18,
            ),
            borderRadius:
                BorderRadius.circular(
              10,
            ),
          ),
          child:
              Text(
            '$_selectedCount',
            style:
                GoogleFonts.dmSans(
              color:
                  AppTheme.tealMint,
              fontSize: 11,
              fontWeight:
                  FontWeight.w800,
            ),
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // HOSTEL SUMMARY
  // ===========================================================================

  Widget _buildHostelSummary() {
    return Container(
      width: double.infinity,
      padding:
          const EdgeInsets.all(
        12,
      ),
      decoration:
          BoxDecoration(
        color:
            const Color(
          0xFF011F21,
        ).withOpacity(
          0.58,
        ),
        borderRadius:
            BorderRadius.circular(
          14,
        ),
        border:
            Border.all(
          color:
              Colors.white
                  .withOpacity(
            0.10,
          ),
        ),
      ),
      child:
          Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration:
                BoxDecoration(
              color:
                  AppTheme.teal
                      .withOpacity(
                0.18,
              ),
              borderRadius:
                  BorderRadius
                      .circular(
                11,
              ),
            ),
            child:
                const Icon(
              Icons.apartment_rounded,
              color:
                  AppTheme.tealMint,
              size: 18,
            ),
          ),

          const SizedBox(
            width: 10,
          ),

          Expanded(
            child:
                Column(
              crossAxisAlignment:
                  CrossAxisAlignment
                      .start,
              children: [
                Text(
                  widget.hostelName,
                  maxLines: 1,
                  overflow:
                      TextOverflow
                          .ellipsis,
                  style:
                      GoogleFonts.dmSans(
                    color:
                        Colors.white,
                    fontSize: 11,
                    fontWeight:
                        FontWeight
                            .w700,
                  ),
                ),

                const SizedBox(
                  height: 2,
                ),

                Text(
                  '${widget.totalRooms} rooms · ${widget.totalCapacity} beds · ${widget.totalFloors} floors',
                  maxLines: 1,
                  overflow:
                      TextOverflow
                          .ellipsis,
                  style:
                      GoogleFonts.dmSans(
                    color: Colors
                        .white
                        .withOpacity(
                      0.46,
                    ),
                    fontSize: 8.3,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // CATEGORY
  // ===========================================================================

  Widget _buildCategory(
    String category,
    List<Map<String, dynamic>>
        facilities,
  ) {
    final color =
        _catColor(category);

    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              width: 29,
              height: 29,
              decoration:
                  BoxDecoration(
                color:
                    color.withOpacity(
                  0.12,
                ),
                borderRadius:
                    BorderRadius.circular(
                  9,
                ),
              ),
              child:
                  Icon(
                _categoryIcon(
                  category,
                ),
                color: color,
                size: 15,
              ),
            ),

            const SizedBox(
              width: 8,
            ),

            Expanded(
              child:
                  Text(
                category,
                style:
                    GoogleFonts.dmSans(
                  color:
                      Colors.white,
                  fontSize: 11,
                  fontWeight:
                      FontWeight
                          .w800,
                ),
              ),
            ),
          ],
        ),

        const SizedBox(
          height: 9,
        ),

        Wrap(
          spacing: 8,
          runSpacing: 8,
          children:
              facilities.map(
            (facility) {
              return _buildFacilityChip(
                facility:
                    facility,
                color: color,
              );
            },
          ).toList(),
        ),
      ],
    );
  }

  // ===========================================================================
  // FACILITY CHIP
  // ===========================================================================

  Widget _buildFacilityChip({
    required Map<String, dynamic>
        facility,
    required Color color,
  }) {
    final selected =
        facility['sel'] == true;

    return InkWell(
      onTap: () {
        setState(() {
          facility['sel'] =
              !selected;

          _error = null;
        });
      },
      borderRadius:
          BorderRadius.circular(
        12,
      ),
      child:
          AnimatedContainer(
        duration:
            const Duration(
          milliseconds: 170,
        ),
        padding:
            const EdgeInsets.symmetric(
          horizontal: 11,
          vertical: 9,
        ),
        decoration:
            BoxDecoration(
          color:
              selected
                  ? color.withOpacity(
                      0.18,
                    )
                  : const Color(
                      0xFF011F21,
                    ).withOpacity(
                      0.52,
                    ),
          borderRadius:
              BorderRadius.circular(
            12,
          ),
          border:
              Border.all(
            color:
                selected
                    ? color.withOpacity(
                        0.70,
                      )
                    : Colors.white
                        .withOpacity(
                        0.10,
                      ),
            width:
                selected
                    ? 1.1
                    : 0.8,
          ),
        ),
        child:
            Row(
          mainAxisSize:
              MainAxisSize.min,
          children: [
            Icon(
              facility['icon']
                  as IconData,
              size: 15,
              color:
                  selected
                      ? color
                      : Colors
                          .white
                          .withOpacity(
                          0.48,
                        ),
            ),

            const SizedBox(
              width: 6,
            ),

            Text(
              facility['label']
                  .toString(),
              style:
                  GoogleFonts.dmSans(
                color:
                    selected
                        ? Colors.white
                        : Colors
                            .white
                            .withOpacity(
                            0.68,
                          ),
                fontSize: 10,
                fontWeight:
                    selected
                        ? FontWeight
                            .w700
                        : FontWeight
                            .w500,
              ),
            ),

            if (selected) ...[
              const SizedBox(
                width: 6,
              ),

              Icon(
                Icons.check_rounded,
                size: 13,
                color: color,
              ),
            ],
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // CUSTOM FACILITIES
  // ===========================================================================

  Widget _buildCustomSection() {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              width: 29,
              height: 29,
              decoration:
                  BoxDecoration(
                color: AppTheme
                    .tealMint
                    .withOpacity(
                  0.12,
                ),
                borderRadius:
                    BorderRadius
                        .circular(
                  9,
                ),
              ),
              child:
                  const Icon(
                Icons
                    .add_business_rounded,
                color:
                    AppTheme.tealMint,
                size: 15,
              ),
            ),

            const SizedBox(
              width: 8,
            ),

            Expanded(
              child:
                  Column(
                crossAxisAlignment:
                    CrossAxisAlignment
                        .start,
                children: [
                  Text(
                    'Additional Facilities',
                    style:
                        GoogleFonts.dmSans(
                      color:
                          Colors.white,
                      fontSize: 11,
                      fontWeight:
                          FontWeight
                              .w800,
                    ),
                  ),

                  Text(
                    'Add something not listed above',
                    style:
                        GoogleFonts.dmSans(
                      color: Colors
                          .white
                          .withOpacity(
                        0.42,
                      ),
                      fontSize: 8,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),

        if (_customList.isNotEmpty) ...[
          const SizedBox(
            height: 10,
          ),

          Wrap(
            spacing: 7,
            runSpacing: 7,
            children:
                _customList.map(
              (name) {
                return Container(
                  padding:
                      const EdgeInsets
                          .fromLTRB(
                    10,
                    7,
                    5,
                    7,
                  ),
                  decoration:
                      BoxDecoration(
                    color: AppTheme
                        .teal
                        .withOpacity(
                      0.16,
                    ),
                    borderRadius:
                        BorderRadius
                            .circular(
                      11,
                    ),
                    border:
                        Border.all(
                      color: AppTheme
                          .tealMint
                          .withOpacity(
                        0.30,
                      ),
                    ),
                  ),
                  child:
                      Row(
                    mainAxisSize:
                        MainAxisSize.min,
                    children: [
                      Text(
                        name,
                        style:
                            GoogleFonts.dmSans(
                          color:
                              Colors.white,
                          fontSize: 9.5,
                          fontWeight:
                              FontWeight
                                  .w600,
                        ),
                      ),

                      const SizedBox(
                        width: 5,
                      ),

                      InkWell(
                        onTap: () {
                          setState(() {
                            _customList
                                .remove(
                              name,
                            );
                          });
                        },
                        borderRadius:
                            BorderRadius
                                .circular(
                          20,
                        ),
                        child:
                            Padding(
                          padding:
                              const EdgeInsets
                                  .all(
                            2,
                          ),
                          child:
                              Icon(
                            Icons
                                .close_rounded,
                            color: AppTheme
                                .tealMint,
                            size: 14,
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              },
            ).toList(),
          ),
        ],

        const SizedBox(
          height: 11,
        ),

        LayoutBuilder(
          builder: (
            context,
            constraints,
          ) {
            // Extra narrow screens par
            // button neeche aa jayega.
            if (constraints
                    .maxWidth <
                260) {
              return Column(
                children: [
                  _customField(),

                  const SizedBox(
                    height: 8,
                  ),

                  SizedBox(
                    width:
                        double.infinity,
                    height: 47,
                    child:
                        ElevatedButton.icon(
                      onPressed:
                          _addCustomFacility,
                      icon:
                          const Icon(
                        Icons
                            .add_rounded,
                        size: 18,
                      ),
                      label:
                          const Text(
                        'Add Facility',
                      ),
                      style:
                          ElevatedButton
                              .styleFrom(
                        backgroundColor:
                            AppTheme.teal,
                        foregroundColor:
                            Colors.white,
                        elevation: 0,
                        shape:
                            RoundedRectangleBorder(
                          borderRadius:
                              BorderRadius
                                  .circular(
                            14,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              );
            }

            return Row(
              children: [
                Expanded(
                  child:
                      _customField(),
                ),

                const SizedBox(
                  width: 8,
                ),

                InkWell(
                  onTap:
                      _addCustomFacility,
                  borderRadius:
                      BorderRadius
                          .circular(
                    14,
                  ),
                  child:
                      Container(
                    width: 48,
                    height: 48,
                    decoration:
                        BoxDecoration(
                      color:
                          AppTheme.teal,
                      borderRadius:
                          BorderRadius
                              .circular(
                        14,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: AppTheme
                              .teal
                              .withOpacity(
                            0.20,
                          ),
                          blurRadius:
                              12,
                          offset:
                              const Offset(
                            0,
                            5,
                          ),
                        ),
                      ],
                    ),
                    child:
                        const Icon(
                      Icons
                          .add_rounded,
                      color:
                          Colors.white,
                      size: 21,
                    ),
                  ),
                ),
              ],
            );
          },
        ),
      ],
    );
  }

  Widget _customField() {
    return AnimatedBuilder(
      animation:
          _customFocus,
      builder: (
        context,
        child,
      ) {
        final focused =
            _customFocus.hasFocus;

        return AnimatedContainer(
          duration:
              const Duration(
            milliseconds: 180,
          ),
          height: 48,
          decoration:
              BoxDecoration(
            // White input box, just like login.
            color:
                Colors.white
                    .withOpacity(
              0.94,
            ),
            borderRadius:
                BorderRadius.circular(
              14,
            ),
            border:
                Border.all(
              color:
                  focused
                      ? AppTheme
                          .tealMint
                      : Colors.white
                          .withOpacity(
                          0.30,
                        ),
              width:
                  focused
                      ? 1.2
                      : 0.8,
            ),
          ),
          child:
              TextField(
            controller:
                _customCtrl,
            focusNode:
                _customFocus,
            textInputAction:
                TextInputAction.done,
            textCapitalization:
                TextCapitalization.words,
            onSubmitted:
                (_) {
              _addCustomFacility();
            },
            style:
                GoogleFonts.dmSans(
              color:
                  const Color(
                0xFF183636,
              ),
              fontSize: 12,
              fontWeight:
                  FontWeight.w500,
            ),
            decoration:
                InputDecoration(
              hintText:
                  'Add another facility',
              hintStyle:
                  GoogleFonts.dmSans(
                color:
                    const Color(
                  0xFF81908F,
                ),
                fontSize: 10.5,
              ),
              prefixIcon:
                  Icon(
                Icons
                    .add_circle_outline_rounded,
                color:
                    const Color(
                  0xFF07595C,
                ),
                size: 18,
              ),
              border:
                  InputBorder.none,
              contentPadding:
                  const EdgeInsets
                      .symmetric(
                horizontal: 13,
                vertical: 13,
              ),
            ),
          ),
        );
      },
    );
  }

  // ===========================================================================
  // ERROR
  // ===========================================================================

  Widget _buildError() {
    return Container(
      width: double.infinity,
      padding:
          const EdgeInsets
              .symmetric(
        horizontal: 12,
        vertical: 10,
      ),
      decoration:
          BoxDecoration(
        color:
            const Color(
          0xFF7F2525,
        ).withOpacity(
          0.42,
        ),
        borderRadius:
            BorderRadius.circular(
          12,
        ),
        border:
            Border.all(
          color:
              const Color(
            0xFFFF7777,
          ).withOpacity(
            0.38,
          ),
        ),
      ),
      child:
          Row(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons
                .error_outline_rounded,
            color:
                Color(
              0xFFFF8A8A,
            ),
            size: 16,
          ),

          const SizedBox(
            width: 8,
          ),

          Expanded(
            child:
                Text(
              _error!,
              style:
                  GoogleFonts.dmSans(
                color:
                    const Color(
                  0xFFFFB5B5,
                ),
                fontSize: 10,
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // NEXT BUTTON
  // ===========================================================================

  Widget _buildNextButton() {
    return SizedBox(
      width: double.infinity,
      height: 50,
      child:
          ElevatedButton(
        onPressed:
            _next,
        style:
            ElevatedButton
                .styleFrom(
          backgroundColor:
              AppTheme.teal,
          foregroundColor:
              Colors.white,
          elevation: 0,
          shape:
              RoundedRectangleBorder(
            borderRadius:
                BorderRadius
                    .circular(
              15,
            ),
          ),
        ),
        child:
            Row(
          mainAxisAlignment:
              MainAxisAlignment
                  .center,
          children: [
            Flexible(
              child: Text(
                'Next: Mess & Food',
                overflow:
                    TextOverflow
                        .ellipsis,
                style:
                    GoogleFonts.dmSans(
                  fontSize:
                      13.5,
                  fontWeight:
                      FontWeight
                          .w800,
                ),
              ),
            ),

            const SizedBox(
              width: 9,
            ),

            Container(
              width: 25,
              height: 25,
              decoration:
                  BoxDecoration(
                color:
                    Colors.white
                        .withOpacity(
                  0.15,
                ),
                shape:
                    BoxShape.circle,
              ),
              child:
                  const Icon(
                Icons
                    .restaurant_rounded,
                size: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }
}